//! Dynamic batching: concurrent requests are merged into one model call.
//!
//! Requests go into a queue. A collector task takes the first job, then keeps taking jobs until the batch holds
//! `max_batch` patches or `max_wait` has passed since the first one, and hands the batch to a pool of worker
//! threads. Each worker owns its own ONNX Runtime session (`Session::run` needs `&mut self`), so workers never
//! contend for a lock during inference. A request larger than `max_batch` (a whole tile) is split into jobs.
//!
//! Memory is bounded by a budget counted in patches (`max_inflight_patches`): a caller takes permits for a job
//! before it lays out the job's inputs, and the permits come back when the job is dropped, so queued and running
//! jobs never hold more than that many patches of inputs, however many requests arrive.

use std::ops::Range;
use std::sync::mpsc as std_mpsc;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use anyhow::{Result, anyhow};
use tokio::sync::{OwnedSemaphorePermit, Semaphore, mpsc, oneshot};

use crate::imaging::PATCH_LEN;
use crate::model::Classifier;

struct Job {
    inputs: Vec<f32>,
    n: usize,
    enqueued: Instant,
    reply: oneshot::Sender<Result<Vec<f32>>>,
    permit: OwnedSemaphorePermit,
}

impl Job {
    /// Free the inputs and return the permits, then answer the caller.
    fn answer(self, result: Result<Vec<f32>>) {
        let Self {
            inputs,
            permit,
            reply,
            ..
        } = self;
        drop((inputs, permit));
        let _ = reply.send(result);
    }
}

struct Batch {
    jobs: Vec<Job>,
    patches: usize,
}

#[derive(Debug, Clone, Copy)]
pub struct BatchConfig {
    pub max_batch: usize,
    pub max_wait: Duration,
    pub workers: usize,
    /// Patches whose inputs may be queued or running at once; each costs `PATCH_LEN` floats.
    pub max_inflight_patches: usize,
}

#[derive(Clone)]
pub struct Batcher {
    tx: mpsc::Sender<Job>,
    budget: Arc<Semaphore>,
    max_batch: usize,
}

impl Batcher {
    /// Start the collector and `workers` threads, each with a classifier from `make`.
    pub fn spawn<F>(config: BatchConfig, make: F) -> Result<Self>
    where
        F: Fn(usize) -> Result<Box<dyn Classifier>>,
    {
        anyhow::ensure!(
            config.max_batch > 0 && config.max_inflight_patches >= config.max_batch,
            "the in-flight budget ({}) must hold at least one full batch ({})",
            config.max_inflight_patches,
            config.max_batch
        );
        // Every queued job holds at least one permit, so the queue cannot fill before the budget does.
        let (tx, rx) = mpsc::channel::<Job>(config.max_inflight_patches);
        let (batch_tx, batch_rx) = std_mpsc::sync_channel::<Batch>(config.workers * 2);
        let batch_rx = Arc::new(Mutex::new(batch_rx));
        for worker in 0..config.workers {
            let mut model = make(worker)?;
            let rx = Arc::clone(&batch_rx);
            std::thread::Builder::new()
                .name(format!("inference-{worker}"))
                .spawn(move || run_worker(model.as_mut(), &rx))?;
        }
        tokio::spawn(collect(rx, batch_tx, config));
        Ok(Self {
            tx,
            budget: Arc::new(Semaphore::new(config.max_inflight_patches)),
            max_batch: config.max_batch,
        })
    }

    #[must_use]
    pub fn max_batch(&self) -> usize {
        self.max_batch
    }

    /// Score `n` patches (at most `max_batch`). Waits for the batch that carries them.
    pub async fn predict(&self, inputs: Vec<f32>, n: usize) -> Result<Vec<f32>> {
        anyhow::ensure!(
            n > 0 && n <= self.max_batch,
            "a job holds 1..={} patches, got {n}",
            self.max_batch
        );
        anyhow::ensure!(
            inputs.len() == n * PATCH_LEN,
            "job has {} values for {n} patches",
            inputs.len()
        );
        let permit = self.acquire(n).await?;
        let reply = self.submit(permit, inputs, n).await?;
        reply.await.map_err(|_| anyhow!("worker dropped the job"))?
    }

    /// Score any number of patches, in order, as jobs of up to `max_batch` that run in parallel.
    ///
    /// `build(range)` lays out the inputs of patches `range`; it is called only once that job holds its permits,
    /// so a request never holds more inputs than the budget allows.
    pub async fn predict_chunked<F>(&self, n: usize, mut build: F) -> Result<Vec<f32>>
    where
        F: FnMut(Range<usize>) -> Vec<f32>,
    {
        anyhow::ensure!(n > 0, "nothing to score");
        let mut replies = Vec::with_capacity(n.div_ceil(self.max_batch));
        for start in (0..n).step_by(self.max_batch) {
            let range = start..n.min(start + self.max_batch);
            let k = range.len();
            let permit = self.acquire(k).await?;
            let inputs = build(range);
            anyhow::ensure!(
                inputs.len() == k * PATCH_LEN,
                "job has {} values for {k} patches",
                inputs.len()
            );
            replies.push(self.submit(permit, inputs, k).await?);
        }
        let mut out = Vec::with_capacity(n);
        for reply in replies {
            out.extend(
                reply
                    .await
                    .map_err(|_| anyhow!("worker dropped the job"))??,
            );
        }
        Ok(out)
    }

    async fn acquire(&self, n: usize) -> Result<OwnedSemaphorePermit> {
        Arc::clone(&self.budget)
            .acquire_many_owned(u32::try_from(n)?)
            .await
            .map_err(|_| anyhow!("batcher stopped"))
    }

    async fn submit(
        &self,
        permit: OwnedSemaphorePermit,
        inputs: Vec<f32>,
        n: usize,
    ) -> Result<oneshot::Receiver<Result<Vec<f32>>>> {
        let (reply, rx) = oneshot::channel();
        self.tx
            .send(Job {
                inputs,
                n,
                enqueued: Instant::now(),
                reply,
                permit,
            })
            .await
            .map_err(|_| anyhow!("batcher stopped"))?;
        metrics::gauge!("pathology_queue_capacity").set(self.tx.capacity() as f64);
        Ok(rx)
    }
}

async fn collect(
    mut rx: mpsc::Receiver<Job>,
    out: std_mpsc::SyncSender<Batch>,
    config: BatchConfig,
) {
    let mut pending: Option<Job> = None;
    loop {
        let first = match pending.take() {
            Some(job) => job,
            None => match rx.recv().await {
                Some(job) => job,
                None => return,
            },
        };
        let deadline = tokio::time::Instant::now() + config.max_wait;
        let mut batch = Batch {
            patches: first.n,
            jobs: vec![first],
        };
        while batch.patches < config.max_batch {
            match tokio::time::timeout_at(deadline, rx.recv()).await {
                Ok(Some(job)) if batch.patches + job.n <= config.max_batch => {
                    batch.patches += job.n;
                    batch.jobs.push(job);
                }
                Ok(Some(job)) => {
                    pending = Some(job); // does not fit: it starts the next batch
                    break;
                }
                Ok(None) | Err(_) => break,
            }
        }
        // Blocking send: when every worker is busy, back-pressure reaches the request queue.
        let out = out.clone();
        if tokio::task::spawn_blocking(move || out.send(batch))
            .await
            .map_or(true, |r| r.is_err())
        {
            return;
        }
    }
}

fn run_worker(model: &mut dyn Classifier, rx: &Mutex<std_mpsc::Receiver<Batch>>) {
    loop {
        let batch = {
            let Ok(guard) = rx.lock() else { return };
            match guard.recv() {
                Ok(b) => b,
                Err(_) => return,
            }
        };
        let started = Instant::now();
        for job in &batch.jobs {
            metrics::histogram!("pathology_queue_wait_seconds")
                .record((started - job.enqueued).as_secs_f64());
        }
        let mut inputs = Vec::with_capacity(batch.patches * PATCH_LEN);
        for job in &batch.jobs {
            inputs.extend_from_slice(&job.inputs);
        }
        let result = model.predict(&inputs, batch.patches);
        metrics::histogram!("pathology_inference_seconds").record(started.elapsed().as_secs_f64());
        metrics::histogram!("pathology_batch_size").record(batch.patches as f64);
        match result {
            Ok(probs) => {
                let mut offset = 0;
                for job in batch.jobs {
                    let part = probs[offset..offset + job.n].to_vec();
                    offset += job.n;
                    job.answer(Ok(part));
                }
            }
            Err(err) => {
                let msg = err.to_string();
                for job in batch.jobs {
                    job.answer(Err(anyhow!("inference failed: {msg}")));
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::FakeClassifier;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// Records the size of every batch it sees.
    struct Counting(Arc<Mutex<Vec<usize>>>);
    impl Classifier for Counting {
        fn predict(&mut self, batch: &[f32], n: usize) -> Result<Vec<f32>> {
            self.0.lock().unwrap().push(n);
            std::thread::sleep(Duration::from_millis(5));
            FakeClassifier.predict(batch, n)
        }
    }

    fn patch(level: f32) -> Vec<f32> {
        vec![level; PATCH_LEN]
    }

    fn config(max_batch: usize, workers: usize, max_inflight_patches: usize) -> BatchConfig {
        BatchConfig {
            max_batch,
            max_wait: Duration::from_millis(1),
            workers,
            max_inflight_patches,
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn concurrent_requests_share_batches_and_get_their_own_answers() {
        let seen = Arc::new(Mutex::new(Vec::new()));
        let s = Arc::clone(&seen);
        let config = BatchConfig {
            max_wait: Duration::from_millis(20),
            ..config(16, 1, 256)
        };
        let batcher = Batcher::spawn(config, move |_| {
            Ok(Box::new(Counting(Arc::clone(&s))) as Box<dyn Classifier>)
        })
        .unwrap();
        let mut handles = Vec::new();
        for i in 0..32 {
            let b = batcher.clone();
            handles.push(tokio::spawn(async move {
                (i, b.predict(patch(i as f32), 1).await.unwrap())
            }));
        }
        for h in handles {
            let (i, p) = h.await.unwrap();
            assert!(
                (p[0] - (1.0 - i as f32 / 255.0)).abs() < 1e-6,
                "request {i} got someone else's answer"
            );
        }
        let sizes = seen.lock().unwrap().clone();
        assert_eq!(sizes.iter().sum::<usize>(), 32);
        assert!(sizes.len() < 32, "requests were never batched: {sizes:?}");
        assert!(sizes.iter().all(|&n| n <= 16));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn predict_chunked_splits_large_requests_and_keeps_order() {
        // 21 patches against a budget of 16: the request has to wait for its own first jobs to finish.
        let batcher = Batcher::spawn(config(8, 2, 16), |_| {
            Ok(Box::new(FakeClassifier) as Box<dyn Classifier>)
        })
        .unwrap();
        let mut built = Vec::new();
        let out = batcher
            .predict_chunked(21, |range| {
                built.push(range.clone());
                range.flat_map(|i| patch(i as f32 * 10.0)).collect()
            })
            .await
            .unwrap();
        assert_eq!(built, [0..8, 8..16, 16..21]);
        assert_eq!(out.len(), 21);
        for (i, p) in out.iter().enumerate() {
            assert!((p - (1.0 - i as f32 * 10.0 / 255.0)).abs() < 1e-5);
        }
        assert_eq!(batcher.budget.available_permits(), 16);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn inputs_are_laid_out_only_when_the_budget_has_room() {
        let batcher = Batcher::spawn(config(4, 1, 4), |_| {
            Ok(Box::new(FakeClassifier) as Box<dyn Classifier>)
        })
        .unwrap();
        let held = batcher.acquire(4).await.unwrap();
        let built = Arc::new(AtomicUsize::new(0));
        let b = batcher.clone();
        let counter = Arc::clone(&built);
        let scoring = tokio::spawn(async move {
            b.predict_chunked(4, |range| {
                counter.fetch_add(1, Ordering::SeqCst);
                vec![0.0; range.len() * PATCH_LEN]
            })
            .await
        });
        tokio::time::sleep(Duration::from_millis(50)).await;
        assert_eq!(built.load(Ordering::SeqCst), 0, "built before admission");
        drop(held);
        assert_eq!(scoring.await.unwrap().unwrap().len(), 4);
        assert_eq!(built.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn a_budget_smaller_than_one_batch_is_refused() {
        let make = |_| Ok(Box::new(FakeClassifier) as Box<dyn Classifier>);
        assert!(Batcher::spawn(config(8, 1, 7), make).is_err());
    }

    #[tokio::test]
    async fn model_errors_reach_every_waiting_request() {
        struct Failing(Arc<AtomicUsize>);
        impl Classifier for Failing {
            fn predict(&mut self, _: &[f32], _: usize) -> Result<Vec<f32>> {
                self.0.fetch_add(1, Ordering::SeqCst);
                Err(anyhow!("onnx runtime exploded"))
            }
        }
        let calls = Arc::new(AtomicUsize::new(0));
        let c = Arc::clone(&calls);
        let batcher = Batcher::spawn(config(4, 1, 8), move |_| {
            Ok(Box::new(Failing(Arc::clone(&c))) as Box<dyn Classifier>)
        })
        .unwrap();
        let err = batcher.predict(patch(1.0), 1).await.unwrap_err();
        assert!(err.to_string().contains("onnx runtime exploded"));
        assert!(
            batcher.predict(patch(1.0), 5).await.is_err(),
            "jobs above max_batch are refused"
        );
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }
}
