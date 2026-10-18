# How this project is run

Scrum, scaled to one developer:

- **Backlog** as data (`docs/agile/backlog.json`), synced to GitHub issues with labels for type, area and story
  points (`scripts/agile/sync_backlog.py`). Every story has acceptance criteria.
- **Sprints** are GitHub milestones with a goal. Work happens on a `sprint-N` branch; commits reference their
  issue (`(#3)`); the sprint ends with one pull request whose description closes the sprint's issues, merged only
  when CI is green.
- **Definition of done**: tests for the behaviour, CI green, numbers in docs traced to a committed result file,
  secret scan clean.
- **Model results** are reported once, on the test split, after every choice (learning rate, epoch, temperature,
  thresholds) was made on the validation split; the selection runs are committed too.
- **Retro** at the end of each sprint, committed here. Work found during a sprint becomes an issue; work that
  needs an account stays in the **Backlog** milestone, labelled `blocked:account`.
