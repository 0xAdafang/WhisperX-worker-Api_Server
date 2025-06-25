class JobStatusStore:
    def __init__(self):
        self.jobs = {}

    def set_status(self, job_id, status):
        self.jobs[job_id] = status

    def get_status(self, job_id):
        return self.jobs.get(job_id, "UNKNOWN")
