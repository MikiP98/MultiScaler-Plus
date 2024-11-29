# Download ImDisk installer
Invoke-WebRequest -Uri "http://www.ltr-data.se/files/imdiskinst.exe" -OutFile "$env:TEMP\imdiskinst.exe"

# Install ImDisk silently
Start-Process -FilePath "$env:TEMP\imdiskinst.exe" -ArgumentList "/S" -Wait

# Check if the installation was successful
if (Get-Command imdisk -ErrorAction SilentlyContinue) {
    Write-Host "ImDisk installed successfully."
    
    # Delete the temporary installer file
    Remove-Item -Path "$env:TEMP\imdiskinst.exe" -Force
} else {
    Write-Host "ImDisk installation failed."
}
