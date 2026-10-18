//! Dynamic batching: concurrent requests are merged into one model call.
//!
//! Requests go into a queue. A collector task takes the first job, then keeps taking jobs until the batch holds
//! `max_batch` patches or `max_wait` has passed since the first one, and hands the batch to a pool of worker
//! threads. Each worker owns its own ONNX Runtime session (`Session::run` needs `&mut self`), so workers never
//! contend for a lock during inference. A job larger than `max_batch` (a whole tile) is split by the caller.

use std::sync::mpsc as std_mpsc;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use anyhow::{Result, anyhow};
use tokio::sync::{mpsc, oneshot};

use crate::imaging::PATCH_LEN;
use crate::model::Classifier;

struct Job {
    inputs: Vec<f32>,
    n: usize,
    enqueued: Instant,
    reply: oneshot::Sender<Result<Vec<f32>>>,
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
    pub queue: usize,
}

#[derive(Clone)]
pub struct Batcher {
    tx: mpsc::Sender<Job>,
    max_batch: usize,
}

impl Batcher {
    /// Start the collector and `workers` threads, each with a classifier from `make`.
    pub fn spawn<F>(config: BatchConfig, make: F) -> Result<Self>
    where
        F: Fn(usize) -> Result<Box<dyn Classifier>>,
    {
        let (tx, rx) = mpsc::channel::<Job>(config.queue);
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
        let (reply, rx) = oneshot::channel();
        self.tx
            .send(Job {
                inputs,
                n,
                enqueued: Instant::now(),
                reply,
            })
            .await
            .map_err(|_| anyhow!("batcher stopped"))?;
        metrics::gauge!("pathology_queue_capacity").set(self.tx.capacity() as f64);
        rx.await.map_err(|_| anyhow!("worker dropped the job"))?
    }

    /// Score any number of patches by splitting them into `max_batch` jobs submitted together.
    pub async fn predict_many(&self, inputs: Vec<f32>, n: usize) -> Result<Vec<f32>> {
        let chunks: Vec<(Vec<f32>, usize)> = inputs
            .chunks(self.max_batch * PATCH_LEN)
            .map(|c| (c.to_vec(), c.len() / PATCH_LEN))
            .collect();
        anyhow::ensure!(
            chunks.iter().map(|c| c.1).sum::<usize>() == n,
            "inputs do not hold {n} patches"
        );
        let futures = chunks.into_iter().map(|(chunk, k)| {
            let batcher = self.clone();
            async move { batcher.predict(chunk, k).await }
        });
        let parts = futures_join(futures).await?;
        Ok(parts.into_iter().flatten().collect())
    }
}

async fn futures_join<I, F>(futs: I) -> Result<Vec<Vec<f32>>>
where
    I: IntoIterator<Item = F>,
    F: std::future::Future<Output = Result<Vec<f32>>> + Send + 'static,
{
    let handles: Vec<_> = futs.into_iter().map(tokio::spawn).collect();
    let mut out = Vec::with_capacity(handles.len());
    for h in handles {
        out.push(h.await.map_err(|e| anyhow!("task failed: {e}"))??);
    }
    Ok(out)
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
                    let _ = job.reply.send(Ok(part));
                }
            }
            Err(err) => {
                let msg = err.to_string();
                for job in batch.jobs {
                    let _ = job.reply.send(Err(anyhow!("inference failed: {msg}")));
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

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn concurrent_requests_share_batches_and_get_their_own_answers() {
        let seen = Arc::new(Mutex::new(Vec::new()));
        let s = Arc::clone(&seen);
        let config = BatchConfig {
            max_batch: 16,
            max_wait: Duration::from_millis(20),
            workers: 1,
            queue: 256,
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
    async fn predict_many_splits_large_jobs_and_keeps_order() {
        let config = BatchConfig {
            max_batch: 8,
            max_wait: Duration::from_millis(1),
            workers: 2,
            queue: 64,
        };
        let batcher = Batcher::spawn(config, |_| {
            Ok(Box::new(FakeClassifier) as Box<dyn Classifier>)
        })
        .unwrap();
        let mut inputs = Vec::new();
        for i in 0..21 {
            inputs.extend(patch(i as f32 * 10.0));
        }
        let out = batcher.predict_many(inputs, 21).await.unwrap();
        assert_eq!(out.len(), 21);
        for (i, p) in out.iter().enumerate() {
            assert!((p - (1.0 - i as f32 * 10.0 / 255.0)).abs() < 1e-5);
        }
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
        let config = BatchConfig {
            max_batch: 4,
            max_wait: Duration::from_millis(1),
            workers: 1,
            queue: 8,
        };
        let batcher = Batcher::spawn(config, move |_| {
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
