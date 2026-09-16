---
applyTo: "{.github/workflows/**,.github/scripts/**,**/conda-recipe/**,docker/**}"
---

# CI and Packaging Instructions for GitHub Copilot

- Fetch a dependency from its canonical upstream, not a personal fork or feature branch. Where a fork is genuinely still required, name in a comment the upstream pull request or issue that will retire it; an unmarked fork is indistinguishable from an abandoned workaround, and collapsing such a conditional as cleanup silently drops the coverage the fork existed for.
