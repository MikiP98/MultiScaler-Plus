C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe -executionpolicy bypass .\content\resources\install_scripts\install_vs_buildtools.ps1
C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe -executionpolicy bypass .\content\resources\install_scripts\InstallImDisk.ps1

@cd content\src

pip install -r requirements.txt

git submodule update --init --recursive --remote

pip uninstall -y opencv-python
pip uninstall -y opencv-contrib-python

pip install opencv-contrib-python

PAUSE