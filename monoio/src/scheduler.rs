use std::{cell::UnsafeCell, collections::VecDeque, marker::PhantomData};

use crate::task::{Schedule, Task};

/// Task priority for the two-tier local scheduler.
///
/// `High` tasks (CQL write handlers, coordinator tasks) are drained before
/// `Background` tasks (compaction, repair, flush) in every reactor iteration.
/// This gives write-path latency priority without starvation — the reactor
/// still drains background work whenever no high-priority tasks are pending.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TaskPriority {
    /// Write-path tasks: CQL handlers, coordinator futures, internode receive.
    High,
    /// Background tasks: compaction, repair, flush, gossip heartbeats.
    Background,
}

pub(crate) struct LocalScheduler {
    pub(crate) priority: TaskPriority,
}

impl Default for LocalScheduler {
    fn default() -> Self {
        Self { priority: TaskPriority::High }
    }
}

impl Schedule for LocalScheduler {
    fn schedule(&self, task: Task<Self>) {
        let p = self.priority;
        crate::runtime::CURRENT.with(|cx| cx.tasks.push(task, p));
    }

    fn yield_now(&self, task: Task<Self>) {
        self.schedule(task);
    }
}

pub(crate) struct TaskQueue {
    // High-priority queue: CQL write tasks, coordinator tasks.
    high: UnsafeCell<VecDeque<Task<LocalScheduler>>>,
    // Background queue: compaction, repair, flush, gossip.
    background: UnsafeCell<VecDeque<Task<LocalScheduler>>>,
    _marker: PhantomData<*const ()>,
}

impl Default for TaskQueue {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for TaskQueue {
    fn drop(&mut self) {
        unsafe {
            let q = &mut *self.high.get();
            while q.pop_front().is_some() {}
            let q = &mut *self.background.get();
            while q.pop_front().is_some() {}
        }
    }
}

impl TaskQueue {
    pub(crate) fn new() -> Self {
        const CAP: usize = 4096;
        Self {
            high: UnsafeCell::new(VecDeque::with_capacity(CAP)),
            background: UnsafeCell::new(VecDeque::with_capacity(CAP / 4)),
            _marker: PhantomData,
        }
    }

    pub(crate) fn len(&self) -> usize {
        unsafe { (*self.high.get()).len() + (*self.background.get()).len() }
    }

    pub(crate) fn is_empty(&self) -> bool {
        unsafe { (*self.high.get()).is_empty() && (*self.background.get()).is_empty() }
    }

    /// Push a task onto the appropriate priority queue.
    pub(crate) fn push(&self, runnable: Task<LocalScheduler>, priority: TaskPriority) {
        unsafe {
            match priority {
                TaskPriority::High => (*self.high.get()).push_back(runnable),
                TaskPriority::Background => (*self.background.get()).push_back(runnable),
            }
        }
    }

    /// Legacy push — defaults to High so existing call sites are unchanged.
    pub(crate) fn push_high(&self, runnable: Task<LocalScheduler>) {
        self.push(runnable, TaskPriority::High);
    }

    /// Pop: High queue first, then Background.
    pub(crate) fn pop(&self) -> Option<Task<LocalScheduler>> {
        unsafe {
            if let Some(t) = (*self.high.get()).pop_front() {
                return Some(t);
            }
            (*self.background.get()).pop_front()
        }
    }

    /// Pop only from the high-priority queue.
    pub(crate) fn pop_high(&self) -> Option<Task<LocalScheduler>> {
        unsafe { (*self.high.get()).pop_front() }
    }

    pub(crate) fn high_is_empty(&self) -> bool {
        unsafe { (*self.high.get()).is_empty() }
    }
}
