param([switch]$Execute)

$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$archive = [System.IO.Path]::GetFullPath((Join-Path $repo 'Obsoleted'))
$comparison = [StringComparison]::OrdinalIgnoreCase
if (-not $archive.StartsWith($repo + [System.IO.Path]::DirectorySeparatorChar, $comparison)) {
    throw 'Archive target is outside the repository.'
}

# Keep selected paper data, its generating scripts and their direct dependencies.
# Shared src/, configs/, CAD, and the future real-robot scripts stay in place.
$keep = @{
    analysis = @(
        'few_shot_axis_adaptation_20261008', 'few_shot_guarded_20261008',
        'initial_pca2_nn_20261008', 'matched_nine_goals_clipped_20261008',
        'paper15_no_contact_floor_on_20260928', 'paper15_paper_bundle_20260928',
        'paper15_unified_tds_corrected_20260928',
        'paper_ff_fb_generalization_20261008',
        'scissors_tds_palm_radius_1p5_20260928',
        'second_axis_feedback_effect_20261006', 'tds_viability_admitted_20260929'
    )
    runs = @('paper15_unified_3m_slew_20260927')
    scripts = @(
        'archive_nonpaper_workspace.ps1', 'build_few_shot_axis_report.py',
        'build_initial_pca_nn_report.py', 'build_matched_nine_report.py',
        'build_paper15_paper_bundle.py', 'build_paper_ff_fb_generalization_report.py',
        'cleanup_pytest_artifacts.ps1', 'evaluate_guarded_task_pls.py',
        'evaluate_initial_pca_nn.py', 'evaluate_initial_pose_conditioning.py',
        'export_push3_cad_reference.py', 'export_training_reward_history.py',
        'few_shot_axis_adaptation.py', 'generate_scissors_drivable_pools.ps1',
        'integrate_scissors_palm_radius_1p5.py', 'render_calibration_trajectories.py',
        'render_matched_representatives.py', 'render_task_initial_states.py',
        'real_experiment', 'replay_matched_nine_ff.py', 'replay_matched_nine_generalization.py',
        'run_matched_nine_campaign.py', 'run_paper15_tds_corrected.py',
        'run_paper15_unified.py', 'run_reach_tds_direct.py',
        'run_scissors_palm_tds_campaign.py', 'run_scissors_retention_v4.py',
        'run_second_axis_feedback.py',
        'simulate_initial_pca_nn.py', 'simulate_initial_pose_axis_selection.py',
        'simulate_initial_pose_conditioning.py', 'summarize_no_contact_tds.py',
        'summarize_paper_one_dof.py', 'validate_guarded_few_shot.py'
    )
    docs = @(
        'README.md', 'claude_external_audit_20261008.md', 'controllers.md',
        'datasets.md', 'experiments.md', 'few_shot_axis_adaptation_20261008.md',
        'full_range_rl_and_synergy_protocol.md', 'initial_pose_authoring.md',
        'paper_one_dof_baseline_20261006.md', 'paper_repository_cleanup_20261011.md',
        'real_robot_integration.md',
        'related_work_pls_synergy.md', 'scissors_palm_sphere_20260926.md',
        'scissors_smooth_control.md', 'scissors_tds_validation.md',
        'second_axis_feedback_experiment_20261006.md', 'spray_lite_training_setup.md',
        'tds_math.md', 'tds_research_workflow.md', 'unified_reward_training.md'
    )
    tests = @(
        'fixtures', '__init__.py', '__pycache__', 'test_action_slew.py', 'test_axis_feedback.py',
        'test_calibration.py', 'test_collector.py', 'test_control.py',
        'test_dependency_boundaries.py', 'test_environments.py', 'test_extensions.py',
        'test_few_shot_axis_adaptation.py', 'test_force.py',
        'test_initial_pca_nn.py', 'test_initial_pose_conditioning.py',
        'test_initial_state_gallery.py', 'test_learning_algorithms.py',
        'test_matched_nine_protocol.py', 'test_reach_tds_direct.py',
        'test_real_tds_trajectory.py', 'test_scissors_smooth.py',
        'test_scissors_tds_evaluation.py', 'test_second_axis_feedback_simulation.py',
        'test_simple_synergy.py', 'test_success_only_tds.py', 'test_synergy.py',
        'test_training_reward_report.py', 'test_paper_report_integrity.py',
        'test_training_validation.py',
        'test_trajectory.py', 'test_transfer.py'
    )
}

$moves = [System.Collections.Generic.List[object]]::new()
foreach ($group in @('analysis', 'runs', 'scripts', 'docs', 'tests')) {
    $sourceRoot = (Resolve-Path -LiteralPath (Join-Path $repo $group)).Path
    if (-not $sourceRoot.StartsWith($repo + [System.IO.Path]::DirectorySeparatorChar, $comparison)) {
        throw "Invalid source root: $sourceRoot"
    }
    $allowed = [System.Collections.Generic.HashSet[string]]::new([StringComparer]::OrdinalIgnoreCase)
    foreach ($name in $keep[$group]) { [void]$allowed.Add($name) }
    foreach ($item in Get-ChildItem -LiteralPath $sourceRoot -Force) {
        if ($allowed.Contains($item.Name)) { continue }
        # These legacy pytest output directories deny even directory listing
        # under the current account. Leave them untouched for an ACL owner.
        if ($group -eq 'analysis' -and $item.Name -like 'pytest_tds_update_*') { continue }
        $source = (Resolve-Path -LiteralPath $item.FullName).Path
        $target = [System.IO.Path]::GetFullPath((Join-Path (Join-Path $archive $group) $item.Name))
        if (-not $source.StartsWith($sourceRoot + [System.IO.Path]::DirectorySeparatorChar, $comparison) -or
            -not $target.StartsWith($archive + [System.IO.Path]::DirectorySeparatorChar, $comparison)) {
            throw "Move path escaped its intended directory: $source -> $target"
        }
        if (Test-Path -LiteralPath $target) { throw "Target already exists: $target" }
        $moves.Add([pscustomobject]@{ Group=$group; Name=$item.Name; Source=$source; Target=$target })
    }
}
foreach ($name in @('.codex_tmp', 'MUJOCO_LOG.TXT')) {
    $sourcePath = Join-Path $repo $name
    if (-not (Test-Path -LiteralPath $sourcePath)) { continue }
    $source = (Resolve-Path -LiteralPath $sourcePath).Path
    $target = [System.IO.Path]::GetFullPath((Join-Path (Join-Path $archive 'root_temp') $name))
    if (-not $source.StartsWith($repo + [System.IO.Path]::DirectorySeparatorChar, $comparison) -or
        -not $target.StartsWith($archive + [System.IO.Path]::DirectorySeparatorChar, $comparison) -or
        (Test-Path -LiteralPath $target)) {
        throw "Invalid temporary move: $source -> $target"
    }
    $moves.Add([pscustomobject]@{ Group='root_temp'; Name=$name; Source=$source; Target=$target })
}

$moves | Group-Object Group | Sort-Object Name | ForEach-Object {
    Write-Output ("{0}: {1} entries" -f $_.Name, $_.Count)
}
if (-not $Execute) {
    Write-Output 'Dry run only. Use -Execute after reviewing the planned entries.'
    $moves | Select-Object Group, Name | Format-Table -AutoSize
    return
}
[void](New-Item -ItemType Directory -Path $archive -Force)
foreach ($move in $moves) {
    [void](New-Item -ItemType Directory -Path (Split-Path -Parent $move.Target) -Force)
    Move-Item -LiteralPath $move.Source -Destination $move.Target -ErrorAction Stop
    if ((Test-Path -LiteralPath $move.Source) -or -not (Test-Path -LiteralPath $move.Target)) {
        throw "Move verification failed: $($move.Source) -> $($move.Target)"
    }
}
$archived = @(
    foreach ($group in @('analysis', 'runs', 'scripts', 'docs', 'tests', 'root_temp')) {
        $directory = Join-Path $archive $group
        if (-not (Test-Path -LiteralPath $directory)) { continue }
        foreach ($item in Get-ChildItem -LiteralPath $directory -Force) {
            [pscustomobject]@{group=$group;name=$item.Name}
        }
    }
)
$manifest = [pscustomobject]@{
    branch = 'chore/paper-only-cleanup-20261011'
    refreshed_at = (Get-Date).ToString('o')
    count = $archived.Count
    skipped_inaccessible = @(
        'analysis/pytest_tds_update_20260925',
        'analysis/pytest_tds_update_20260925b',
        'analysis/pytest_tds_update_20260925c',
        'analysis/pytest_tds_update_20260925d',
        '.pytest_cache'
    )
    moves = $archived
}
$manifest | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $archive 'move_manifest.json') -Encoding UTF8
Write-Output ("Archived {0} entries total in {1}" -f $archived.Count, $archive)
