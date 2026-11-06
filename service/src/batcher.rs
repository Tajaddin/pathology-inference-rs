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
//!
//! Overload is refused rather than queued. A request first reserves all of its patches among
//! `max_pending_patches`, the work admitted but not yet answered, and fails with [`Overloaded`] if there is no room
//! within `admit_timeout`. An admitted request therefore waits behind at most that much work, whatever its size,
//! and is never cut off halfway. Jobs whose caller has gone (a timeout or a closed connection) are dropped
//! unscored.

use std::ops::Range;
use std::sync::atomic::{AtomicUsize, Ordering};
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
    /// Patches of admitted requests not yet answered; no request may be larger.
    pub max_pending_patches: usize,
    /// How long a new request may wait for room among the pending patches before it is refused.
    pub admit_timeout: Duration,
}

/// There was no room for a new request among the pending patches within `admit_timeout`.
#[derive(Debug, thiserror::Error)]
#[error("the service is at capacity; retry shortly")]
pub struct Overloaded;

/// Room reserved for one request among the pending patches; dropping it gives the room back.
pub struct Admission {
    room: OwnedSemaphorePermit,
    n: usize,
}

#[derive(Clone)]
pub struct Batcher {
    tx: mpsc::Sender<Job>,
    budget: Arc<Semaphore>,
    pending: Arc<Semaphore>,
    max_pending: usize,
    max_batch: usize,
    admit_timeout: Duration,
    workers: usize,
    running: Arc<AtomicUsize>,
}

/// Counts a worker thread as running until it exits, including by a panic.
struct Running(Arc<AtomicUsize>);

impl Drop for Running {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::AcqRel);
    }
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
        anyhow::ensure!(
            config.max_pending_patches >= config.max_batch,
            "the pending limit ({}) must hold at least one full batch ({})",
            config.max_pending_patches,
            config.max_batch
        );
        // Every queued job holds at least one permit, so the queue cannot fill before the budget does.
        let (tx, rx) = mpsc::channel::<Job>(config.max_inflight_patches);
        let (batch_tx, batch_rx) = std_mpsc::sync_channel::<Batch>(config.workers * 2);
        let batch_rx = Arc::new(Mutex::new(batch_rx));
        let running = Arc::new(AtomicUsize::new(0));
        for worker in 0..config.workers {
            let mut model = make(worker)?;
            let rx = Arc::clone(&batch_rx);
            running.fetch_add(1, Ordering::AcqRel);
            let guard = Running(Arc::clone(&running));
            std::thread::Builder::new()
                .name(format!("inference-{worker}"))
                .spawn(move || {
                    let _guard = guard;
                    run_worker(model.as_mut(), &rx);
                })?;
        }
        tokio::spawn(collect(rx, batch_tx, config));
        Ok(Self {
            tx,
            budget: Arc::new(Semaphore::new(config.max_inflight_patches)),
            pending: Arc::new(Semaphore::new(config.max_pending_patches)),
            max_pending: config.max_pending_patches,
            max_batch: config.max_batch,
            admit_timeout: config.admit_timeout,
            workers: config.workers,
            running,
        })
    }

    /// Every worker thread is still running.
    #[must_use]
    pub fn healthy(&self) -> bool {
        self.running.load(Ordering::Acquire) == self.workers
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
        let _admitted = self.admit(n).await?;
        let permit = self.acquire(n).await?;
        let reply = self.submit(permit, inputs, n).await?;
        reply.await.map_err(|_| anyhow!("worker dropped the job"))?
    }

    /// Score the patches of an admitted request, in order, as jobs of up to `max_batch` that run in parallel.
    ///
    /// `build(range)` lays out the inputs of patches `range`; it is called only once that job holds its permits,
    /// so a request never holds more inputs than the budget allows. Dropping the returned future abandons the
    /// jobs it has queued and gives the room back.
    pub async fn predict_chunked<F>(&self, admission: Admission, mut build: F) -> Result<Vec<f32>>
    where
        F: FnMut(Range<usize>) -> Vec<f32>,
    {
        let Admission { room: _room, n } = admission;
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

    /// Reserve room for a request of `n` patches among the pending work, or fail with [`Overloaded`] after
    /// `admit_timeout`. Callers admit a request before decoding it, so one that is refused costs nothing.
    pub async fn admit(&self, n: usize) -> Result<Admission> {
        anyhow::ensure!(
            n > 0 && n <= self.max_pending,
            "a request holds 1..={} patches, got {n}",
            self.max_pending
        );
        let started = Instant::now();
        let room = Arc::clone(&self.pending).acquire_many_owned(u32::try_from(n)?);
        let room = tokio::time::timeout(self.admit_timeout, room).await;
        metrics::histogram!("pathology_admission_wait_seconds")
            .record(started.elapsed().as_secs_f64());
        let room = room
            .map_err(|_| anyhow::Error::new(Overloaded))?
            .map_err(|_| anyhow!("batcher stopped"))?;
        Ok(Admission { room, n })
    }

    async fn acquire(&self, n: usize) -> Result<OwnedSemaphorePermit> {
        let started = Instant::now();
        let permit = Arc::clone(&self.budget)
            .acquire_many_owned(u32::try_from(n)?)
            .await
            .map_err(|_| anyhow!("batcher stopped"));
        metrics::histogram!("pathology_budget_wait_seconds")
            .record(started.elapsed().as_secs_f64());
        permit
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
        if first.reply.is_closed() {
            continue; // the caller has gone; dropping the job returns its permits
        }
        let deadline = tokio::time::Instant::now() + config.max_wait;
        let mut batch = Batch {
            patches: first.n,
            jobs: vec![first],
        };
        while batch.patches < config.max_batch {
            match tokio::time::timeout_at(deadline, rx.recv()).await {
                Ok(Some(job)) if job.reply.is_closed() => {}
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
        let mut batch = {
            let Ok(guard) = rx.lock() else { return };
            match guard.recv() {
                Ok(b) => b,
                Err(_) => return,
            }
        };
        // Callers can give up while their batch waits for a worker.
        batch.jobs.retain(|job| !job.reply.is_closed());
        if batch.jobs.is_empty() {
            continue;
        }
        batch.patches = batch.jobs.iter().map(|job| job.n).sum();
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
            Ok(probs) if probs.len() != batch.patches => {
                let msg = format!(
                    "model returned {} outputs for {}",
                    probs.len(),
                    batch.patches
                );
                for job in batch.jobs {
                    job.answer(Err(anyhow!("inference failed: {msg}")));
                }
            }
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
            max_pending_patches: 1024,
            admit_timeout: Duration::from_secs(1),
        }
    }

    /// Counts its batches and takes 10 ms over each.
    struct Slow(Arc<AtomicUsize>);
    impl Classifier for Slow {
        fn predict(&mut self, batch: &[f32], n: usize) -> Result<Vec<f32>> {
            self.0.fetch_add(1, Ordering::SeqCst);
            std::thread::sleep(Duration::from_millis(10));
            FakeClassifier.predict(batch, n)
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
        let admission = batcher.admit(21).await.unwrap();
        let out = batcher
            .predict_chunked(admission, |range| {
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
            let admission = b.admit(4).await?;
            b.predict_chunked(admission, |range| {
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

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn abandoned_requests_are_not_scored() {
        let calls = Arc::new(AtomicUsize::new(0));
        let c = Arc::clone(&calls);
        let batcher = Batcher::spawn(config(8, 1, 1024), move |_| {
            Ok(Box::new(Slow(Arc::clone(&c))) as Box<dyn Classifier>)
        })
        .unwrap();
        // 20 jobs of 8 patches at 10 ms each; the caller gives up after 25 ms.
        let scoring = async {
            let admission = batcher.admit(160).await?;
            batcher
                .predict_chunked(admission, |range| vec![1.0; range.len() * PATCH_LEN])
                .await
        };
        assert!(
            tokio::time::timeout(Duration::from_millis(25), scoring)
                .await
                .is_err()
        );
        let at_cancel = calls.load(Ordering::SeqCst);
        tokio::time::sleep(Duration::from_millis(300)).await;
        let after = calls.load(Ordering::SeqCst);
        assert!(
            after <= at_cancel + 1,
            "{at_cancel} batches had run when the caller left, {after} in the end"
        );
        assert_eq!(batcher.budget.available_permits(), 1024);
        assert_eq!(batcher.pending.available_permits(), 1024);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn requests_are_refused_when_the_pending_work_stays_full() {
        let batcher = Batcher::spawn(
            BatchConfig {
                max_pending_patches: 8,
                admit_timeout: Duration::from_millis(20),
                ..config(8, 1, 8)
            },
            |_| Ok(Box::new(FakeClassifier) as Box<dyn Classifier>),
        )
        .unwrap();
        let held = batcher.admit(8).await.unwrap();
        let started = Instant::now();
        let err = batcher.predict(patch(1.0), 1).await.unwrap_err();
        assert!(err.is::<Overloaded>(), "{err}");
        assert!(started.elapsed() < Duration::from_millis(500));
        let err = batcher.admit(1).await.err().unwrap();
        assert!(err.is::<Overloaded>(), "{err}");
        drop(held);
        assert_eq!(batcher.predict(patch(1.0), 1).await.unwrap().len(), 1);
    }

    #[tokio::test]
    async fn a_short_model_output_fails_the_batch_not_the_worker() {
        struct Short;
        impl Classifier for Short {
            fn predict(&mut self, batch: &[f32], n: usize) -> Result<Vec<f32>> {
                let mut probs = FakeClassifier.predict(batch, n)?;
                probs.pop();
                Ok(probs)
            }
        }
        let batcher = Batcher::spawn(config(4, 1, 8), |_| {
            Ok(Box::new(Short) as Box<dyn Classifier>)
        })
        .unwrap();
        for _ in 0..2 {
            let err = batcher.predict(patch(1.0), 1).await.unwrap_err();
            assert!(
                err.to_string().contains("returned 0 outputs for 1"),
                "{err}"
            );
        }
        assert!(batcher.healthy());
    }

    #[tokio::test]
    async fn a_worker_that_dies_makes_the_batcher_unhealthy() {
        struct Panicking;
        impl Classifier for Panicking {
            fn predict(&mut self, _: &[f32], _: usize) -> Result<Vec<f32>> {
                panic!("bug in the model wrapper")
            }
        }
        let batcher = Batcher::spawn(config(4, 2, 8), |_| {
            Ok(Box::new(Panicking) as Box<dyn Classifier>)
        })
        .unwrap();
        assert!(batcher.healthy());
        assert!(batcher.predict(patch(1.0), 1).await.is_err());
        tokio::time::sleep(Duration::from_millis(50)).await;
        assert!(!batcher.healthy());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn an_admitted_request_is_not_held_back_by_later_arrivals() {
        let calls = Arc::new(AtomicUsize::new(0));
        let c = Arc::clone(&calls);
        let batcher = Batcher::spawn(
            BatchConfig {
                max_pending_patches: 64,
                admit_timeout: Duration::from_millis(20),
                ..config(8, 1, 8)
            },
            move |_| Ok(Box::new(Slow(Arc::clone(&c))) as Box<dyn Classifier>),
        )
        .unwrap();
        let b = batcher.clone();
        // Eight jobs at 10 ms each, through a budget that holds only one of them at a time.
        let large = tokio::spawn(async move {
            let admission = b.admit(64).await?;
            b.predict_chunked(admission, |range| vec![1.0; range.len() * PATCH_LEN])
                .await
        });
        tokio::time::sleep(Duration::from_millis(5)).await;
        let err = batcher.predict(patch(1.0), 1).await.unwrap_err();
        assert!(err.is::<Overloaded>(), "{err}");
        assert_eq!(large.await.unwrap().unwrap().len(), 64);
        assert_eq!(calls.load(Ordering::SeqCst), 8);
        assert!(
            batcher.admit(65).await.is_err(),
            "larger than the pending limit"
        );
    }

    /// The `_sum` of a histogram in a Prometheus rendering, in seconds.
    fn seconds_waited(text: &str, name: &str) -> f64 {
        let prefix = format!("{name}_sum ");
        text.lines()
            .find_map(|line| line.strip_prefix(prefix.as_str()))
            .map_or(0.0, |v| v.parse().unwrap())
    }

    #[tokio::test]
    async fn waits_for_admission_and_for_the_budget_are_measured() {
        let recorder = metrics_exporter_prometheus::PrometheusBuilder::new().build_recorder();
        let prometheus = recorder.handle();
        let _local = metrics::set_default_local_recorder(&recorder);
        let batcher = Batcher::spawn(
            BatchConfig {
                max_pending_patches: 4,
                admit_timeout: Duration::from_millis(30),
                ..config(4, 1, 4)
            },
            |_| Ok(Box::new(FakeClassifier) as Box<dyn Classifier>),
        )
        .unwrap();
        let room = batcher.admit(4).await.unwrap();
        assert!(batcher.admit(1).await.is_err());
        drop(room);
        let budget = batcher.acquire(4).await.unwrap();
        let b = batcher.clone();
        let scoring = tokio::spawn(async move { b.predict(patch(1.0), 1).await });
        tokio::time::sleep(Duration::from_millis(50)).await;
        drop(budget);
        scoring.await.unwrap().unwrap();

        let text = prometheus.render();
        assert!(
            seconds_waited(&text, "pathology_admission_wait_seconds") >= 0.03,
            "{text}"
        );
        assert!(
            seconds_waited(&text, "pathology_budget_wait_seconds") >= 0.05,
            "{text}"
        );
        assert!(!text.contains("pathology_queue_capacity"), "{text}");
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
