import json
from pathlib import Path


class ExperienceStore:
    def __init__(self, path, load_existing=False):
        self.path = Path(path)
        self.experiences = []

        if load_existing and self.path.exists():
            with open(self.path) as file:
                self.experiences = json.load(file)

    def add(self, experience):
        self.experiences.append(experience)

    def upsert(self, experience):
        for i, item in enumerate(self.experiences):
            if item["query"] == experience["query"]:
                self.experiences[i] = experience
                return

        self.experiences.append(experience)

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)

        with open(self.path, "w") as file:
            json.dump(self.experiences, file, indent=2)

    def get_all(self):
        return self.experiences