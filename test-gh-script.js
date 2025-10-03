// Test the GitHub script logic locally
const { execSync } = require('child_process');

// Simulate the GitHub API calls
console.log('🧪 Testing GitHub script logic...');

// Get open PRs (using gh CLI)
try {
  const pullsOutput = execSync('gh pr list --state open --json number,title,headRefName,mergeable,author', { encoding: 'utf8' });
  const pulls = JSON.parse(pullsOutput);

  console.log(`📋 Found ${pulls.length} open PRs:`);

  const approvedPRs = [];

  for (const pr of pulls) {
    console.log(`\n📄 PR #${pr.number}: ${pr.title}`);
    console.log(`   Author: ${pr.author.login}`);
    console.log(`   Branch: ${pr.headRefName}`);
    console.log(`   Mergeable: ${pr.mergeable}`);

    // Check status checks (no approval needed, just passing checks)
    try {
      const statusOutput = execSync(`gh pr view ${pr.number} --json statusCheckRollup`, { encoding: 'utf8' });
      const statusData = JSON.parse(statusOutput);

      const deploySiteCheck = statusData.statusCheckRollup.find(check => check.name === 'deploy');
      const filterFilesCheck = statusData.statusCheckRollup.find(check => check.name === 'Detect what files changed');

      const deploySitePassing = deploySiteCheck && deploySiteCheck.conclusion === 'SUCCESS';
      const filterFilesPassing = filterFilesCheck && filterFilesCheck.conclusion === 'SUCCESS';

      console.log(`   Deploy site check: ${deploySiteCheck ? deploySiteCheck.conclusion : 'NOT_FOUND'}`);
      console.log(`   Filter-files check: ${filterFilesCheck ? filterFilesCheck.conclusion : 'NOT_FOUND'}`);

      if (deploySitePassing && filterFilesPassing) {
        approvedPRs.push({
          number: pr.number,
          title: pr.title,
          headRefName: pr.headRefName
        });
        console.log(`   ✅ Qualifies for deployment!`);
      } else {
        console.log(`   ❌ Does not qualify (missing passing checks)`);
      }
    } catch (e) {
      console.log(`   ⚠️  Could not get status checks: ${e.message}`);
    }
  }

  console.log(`\n🎯 Final result: ${approvedPRs.length} PRs qualify for deployment:`);
  approvedPRs.forEach(pr => {
    console.log(`   - PR #${pr.number}: ${pr.title}`);
  });

} catch (e) {
  console.error('❌ Error:', e.message);
}