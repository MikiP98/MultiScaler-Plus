# Define the URL for Visual Studio Build Tools installer
$vsBuildToolsUrl = "https://aka.ms/vs/17/release/vs_buildtools.exe"

# Define the path where the installer will be downloaded
$tempFilePath = "$env:TEMP\vs_buildtools.exe"

# Download Visual Studio Build Tools installer
Invoke-WebRequest -Uri $vsBuildToolsUrl -OutFile $tempFilePath

# Install Visual Studio Build Tools with specified options (removed --downloadThenInstall from the argument list)
Start-Process -FilePath $tempFilePath -ArgumentList "--norestart --passive --includeRecommended --add Microsoft.VisualStudio.Workload.NativeDesktop --add Microsoft.VisualStudio.Workload.VCTools --add Microsoft.VisualStudio.Workload.MSBuildTools" -Wait

# Check if the installation was successful
if ($?) {
    Write-Host "Visual Studio Build Tools installed successfully."

    # Delete the temporary installer file
    Remove-Item -Path $tempFilePath -Force
} else {
    Write-Host "Visual Studio Build Tools installation failed."
}
