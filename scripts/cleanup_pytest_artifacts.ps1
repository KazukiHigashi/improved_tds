param([switch]$Delete)
$ErrorActionPreference = 'Stop'
$cleanWorkspace = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
$cleanRoots = @((Get-Item -LiteralPath $cleanWorkspace)) + @(Get-ChildItem -LiteralPath $cleanWorkspace -Directory -Force | Where-Object {-not $_.Name.StartsWith('.')})
$cleanResults = @()
foreach ($cleanRoot in $cleanRoots) {
    $cleanCandidates = @(Get-ChildItem -LiteralPath $cleanRoot.FullName -Directory -Force | Where-Object {
        $_.Name -eq '.pytest_cache' -or $_.Name.StartsWith('.pytest_tmp') -or $_.Name.StartsWith('.test_') -or $_.Name -eq '.codex_test_reward_common'
    })
    foreach ($cleanDir in $cleanCandidates) {
        $cleanPath = [IO.Path]::GetFullPath($cleanDir.FullName)
        if (-not $cleanPath.StartsWith($cleanWorkspace + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) { throw "Outside workspace: $cleanPath" }
        if ([IO.Path]::GetDirectoryName($cleanPath) -ne $cleanRoot.FullName) { throw "Unexpected parent: $cleanPath" }
        $cleanRecord = [ordered]@{path=$cleanPath;status='pending';files=0;bytes=0}
        try {
            # Workspace itself is a non-Git container; child repositories are
            # checked independently. Never infer untracked from a Git error.
            $cleanWorkspaceGit = Join-Path $cleanWorkspace '.git'
            $cleanWorkspaceHasGit = (Test-Path -LiteralPath $cleanWorkspaceGit) -and (@(Get-ChildItem -LiteralPath $cleanWorkspaceGit -Force).Count -gt 0)
            if ($cleanRoot.FullName -ne $cleanWorkspace -or $cleanWorkspaceHasGit) {
                $cleanTracked = @(git -C $cleanRoot.FullName ls-files -- $cleanDir.Name)
                if ($LASTEXITCODE -ne 0) {throw 'Unable to check tracked files'}
                if ($cleanTracked.Count -gt 0) {throw 'Contains tracked files; retained'}
            }
            # Inspect without ever following junctions or symbolic links.
            $cleanStack = New-Object 'System.Collections.Generic.Stack[System.IO.DirectoryInfo]'
            $cleanStack.Push($cleanDir)
            while ($cleanStack.Count -gt 0) {
                $cleanNode = $cleanStack.Pop()
                if ($cleanNode.Attributes -band [IO.FileAttributes]::ReparsePoint) {throw 'Reparse point; retained'}
                foreach ($cleanItem in Get-ChildItem -LiteralPath $cleanNode.FullName -Force) {
                    if ($cleanItem.Attributes -band [IO.FileAttributes]::ReparsePoint) {throw 'Nested reparse point; retained'}
                    if ($cleanItem.PSIsContainer) {$cleanStack.Push($cleanItem)}
                    else {$cleanRecord.files++;$cleanRecord.bytes += $cleanItem.Length}
                }
            }
            if ($Delete) {
                Remove-Item -LiteralPath $cleanPath -Recurse -Force
                $cleanRecord.status='deleted'
            } else {$cleanRecord.status='safe_to_delete'}
        } catch {
            $cleanRecord.status='retained'
            $cleanRecord.error=$_.Exception.Message
        }
        $cleanResults += [pscustomobject]$cleanRecord
    }
}
$cleanResults | ConvertTo-Json -Depth 3
