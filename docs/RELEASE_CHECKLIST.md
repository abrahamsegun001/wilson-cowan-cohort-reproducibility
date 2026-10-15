# Public-release checklist

Before the journal submission is finalized:

- [ ] Run the complete pipeline or confirm the frozen production outputs on Narval.
- [ ] Run `scripts/collect_release_outputs.sh ~/scratch/gpinn_project`.
- [ ] Run `python scripts/validate_release.py` and obtain `RELEASE VALIDATION PASSED`.
- [ ] Confirm that no raw animal-imaging files, credentials, cluster logs, or large checkpoints are committed.
- [ ] Confirm `config.yaml` matches the manuscript.
- [ ] Confirm the author list in `CITATION.cff` matches the final manuscript authorship.
- [ ] Push the consolidated repository to GitHub.
- [ ] Make the repository public before submission if required by the journal.
- [ ] Create a versioned release (suggested tag: `v1.0.0`).
- [ ] Archive the tagged release in Zenodo and obtain a DOI.
- [ ] Replace the manuscript's future-tense Code Availability wording with the final public URL and archival DOI.
- [ ] Record the commit hash used for the submitted manuscript.
