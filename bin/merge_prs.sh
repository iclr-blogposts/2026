#!/usr/bin/env bash
set -euo pipefail

OWNER="iclr-blogposts"   # replace with your repo owner
REPO="2026"     # replace with your repo name
BASE_BRANCH="main" # or "master"

# 1. Make sure we’re up to date
git fetch origin

# 2. Start fresh _tmp_deploy branch
git checkout "$BASE_BRANCH"
git pull origin "$BASE_BRANCH"
git branch -D _tmp_deploy 2>/dev/null || true
git checkout -b _tmp_deploy

# 3. Get successful PRs
prs=$(gh pr list --repo "$OWNER/$REPO" --state open -L 1000000 --json number \
  | jq -r '.[].number')

for pr in $prs; do
  # Get PR status using the checks API (works better with forks)
  state=$(gh pr checks "$pr" --repo "$OWNER/$REPO" --json state --jq '.[0].state // "UNKNOWN"' 2>/dev/null || echo "UNKNOWN")
  
  # If checks API doesn't work, try the commit status API
  if [ "$state" = "UNKNOWN" ] || [ "$state" = "null" ]; then
    head_sha=$(gh pr view "$pr" --repo "$OWNER/$REPO" --json headRefOid --jq '.headRefOid')
    state=$(gh api repos/"$OWNER"/"$REPO"/commits/"$head_sha"/status --jq '.state // "UNKNOWN"' 2>/dev/null || echo "UNKNOWN")
  fi

  if [ "$state" = "SUCCESS" ] || [ "$state" = "success" ]; then
    echo "✅ Merging PR #$pr"
    
    # 4. Fetch head branch (works even if from fork)
    pr_info=$(gh api repos/$OWNER/$REPO/pulls/$pr --jq '{owner: .head.repo.owner.login, repo: .head.repo.name, ref: .head.ref}')
    fork_owner=$(echo "$pr_info" | jq -r '.owner')
    fork_repo=$(echo "$pr_info" | jq -r '.repo')
    fork_ref=$(echo "$pr_info" | jq -r '.ref')

    # Clean up any existing local branch reference
    git branch -D "$fork_owner-$fork_ref" 2>/dev/null || true
    
    git fetch https://github.com/$fork_owner/$fork_repo.git "$fork_ref:$fork_owner-$fork_ref"
    git merge --no-edit --allow-unrelated-histories "$fork_owner-$fork_ref"
  else
    echo "❌ Skipping PR #$pr (state=$state)"
  fi
done

echo "🎉 All successful PRs merged into _tmp_deploy"
