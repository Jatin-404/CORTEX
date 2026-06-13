from celery import Celery

from cortex.settings import settings

celery_app = Celery(
    "cortex",
    broker=settings.celery_broker,
    backend=settings.celery_backend,
    include=["cortex.worker.tasks"],
)

celery_app.conf.update(
    task_default_queue=settings.celery_task_default_queue,
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_track_started=True,
    worker_prefetch_multiplier=1,
    task_acks_late=True,
    task_reject_on_worker_lost=True,
)
