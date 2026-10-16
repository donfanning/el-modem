; El-Modem Windows installer (Inno Setup 6.1+).
; Does NOT contain Microsoft's runtime. If the Visual C++ Redistributable 2015-2022 (x64) is missing or
; too old, it is downloaded from Microsoft at install time, checked for a valid Microsoft Authenticode
; signature, then run. See lab/WINDOWS_ATTRIBUTION_2026_10_13.md.
;
; iscc /DSourceDir=<extracted candidate archive> /DAppVersion=0.1.0-rc1 /DOutputDir=<dir> el-modem.iss
#ifndef SourceDir
  #error SourceDir (extracted el-modem-*-windows-x86_64 archive) is required
#endif
#ifndef AppVersion
  #define AppVersion "0.0.0"
#endif
#ifndef OutputDir
  #define OutputDir "."
#endif
; DriverDir is the staged, signed virtual-COM driver payload
; (build_driver_cert.ps1 + sign_driver.ps1 + the WDK build output). Absent for
; builds produced without a signed driver, in which case nothing driver-related
; is installed and no certificate is touched.
#ifndef DriverDir
  #define DriverDir ""
#endif

[Setup]
AppId={{6C1D55D0-6B0C-4E4F-9A55-4C0A1E5B7A21}
AppName=El-Modem
AppVersion={#AppVersion}
AppPublisher=El-Modem project
DefaultDirName={autopf}\El-Modem
DefaultGroupName=El-Modem
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
MinVersion=10.0
PrivilegesRequired=admin
OutputDir={#OutputDir}
OutputBaseFilename=el-modem-{#AppVersion}-windows-x86_64-setup
Compression=lzma2
SolidCompression=yes
LicenseFile={#SourceDir}\LICENSE
UninstallDisplayName=El-Modem
DisableProgramGroupPage=yes

[Files]
Source: "{#SourceDir}\el-modem.exe"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#SourceDir}\LICENSE"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#SourceDir}\NOTICE"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#SourceDir}\CREDITS.md"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#SourceDir}\README.txt"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#SourceDir}\MANIFEST.json"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#SourceDir}\docs\*"; DestDir: "{app}\docs"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "{#SourceDir}\third-party\*"; DestDir: "{app}\third-party"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "{#SourceDir}\qualification\*"; DestDir: "{app}\qualification"; Flags: ignoreversion recursesubdirs createallsubdirs skipifsourcedoesntexist

[Dirs]
; The signed driver payload and the scripts that install and remove it. The
; signing key is NOT here: only the public .cer ever ships. The directory is
; always created; install_driver.ps1 reports clearly when there is no payload.
Name: "{app}\driver"
Source: "{#DriverDir}\*"; DestDir: "{app}\driver"; Flags: ignoreversion recursesubdirs createallsubdirs skipifsourcedoesntexist
Source: "install_driver.ps1"; DestDir: "{app}\driver"; Flags: ignoreversion skipifsourcedoesntexist
Source: "uninstall_driver.ps1"; DestDir: "{app}\driver"; Flags: ignoreversion skipifsourcedoesntexist

[Icons]
; El-Modem is a terminal application: the shortcut opens it in a console window.
Name: "{group}\El-Modem"; Filename: "{app}\el-modem.exe"
Name: "{group}\Uninstall El-Modem"; Filename: "{uninstallexe}"

[Code]
const
  VCRedistUrl = 'https://aka.ms/vs/17/release/vc_redist.x64.exe';
  VCRegistryKey = 'SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64';
  MinimumMinor = 40;   { 14.40 = Visual Studio 2022 17.10; the runtime the frozen build was checked with }
  WizardNoDriver = '0'; { set to '1' when the user declines the test-signed driver }

function VCRuntimeCurrent: Boolean;
var
  Installed, Major, Minor: Cardinal;
begin
  Result := False;
  if RegQueryDWordValue(HKLM64, VCRegistryKey, 'Installed', Installed) and (Installed = 1) and
     RegQueryDWordValue(HKLM64, VCRegistryKey, 'Major', Major) and
     RegQueryDWordValue(HKLM64, VCRegistryKey, 'Minor', Minor) then
    Result := (Major > 14) or ((Major = 14) and (Minor >= MinimumMinor));
end;

function SignedByMicrosoft(const FileName: String): Boolean;
var
  Code: Integer;
  Script, ScriptFile, Report, Outcome: String;
  Raw: AnsiString;
begin
  { The download URL is a moving target, so a pinned hash is impossible. Require a valid
    Authenticode signature whose signer is Microsoft Corporation instead. The check runs from a
    script file (no command-line quoting), with the machine module path (a PowerShell 7 parent otherwise leaks its modules into Windows PowerShell 5.1), and the verdict is logged so a refusal can be diagnosed. }
  ScriptFile := ExpandConstant('{tmp}\check-signature.ps1');
  Report := ExpandConstant('{tmp}\signature-check.txt');
  Script :=
    'param([string]$File, [string]$Report)' + #13#10 +
    '$env:PSModulePath = [Environment]::GetEnvironmentVariable("PSModulePath", "Machine")' + #13#10 +
    '$ErrorActionPreference = "Stop"' + #13#10 +
    'try {' + #13#10 +
    '  $s = Get-AuthenticodeSignature -LiteralPath $File' + #13#10 +
    '  $subject = "(none)"' + #13#10 +
    '  if ($s.SignerCertificate) { $subject = $s.SignerCertificate.Subject }' + #13#10 +
    '  "$($s.Status)|$subject" | Out-File -LiteralPath $Report -Encoding ascii' + #13#10 +
    '  if ($s.Status -eq "Valid" -and $subject -match "O=Microsoft Corporation") { exit 0 } else { exit 1 }' + #13#10 +
    '} catch { "error|$($_.Exception.Message)" | Out-File -LiteralPath $Report -Encoding ascii; exit 2 }' + #13#10;
  SaveStringToFile(ScriptFile, Script, False);
  Result := Exec(ExpandConstant('{sys}\WindowsPowerShell\v1.0\powershell.exe'),
                 '-NoProfile -NonInteractive -ExecutionPolicy Bypass -File "' + ScriptFile + '" "' + FileName + '" "' + Report + '"',
                 '', SW_HIDE, ewWaitUntilTerminated, Code) and (Code = 0);
  Outcome := '(no report)';
  if LoadStringFromFile(Report, Raw) then Outcome := Trim(String(Raw));
  Log('Runtime installer signature check: exit ' + IntToStr(Code) + ', ' + Outcome);
end;

function InstallVCRuntime(var NeedsRestart: Boolean): String;
var
  Target: String;
  Code: Integer;
  Started: Boolean;
begin
  Result := '';
  try
    DownloadTemporaryFile(VCRedistUrl, 'vc_redist.x64.exe', '', nil);
  except
    Result := 'Could not download the Microsoft Visual C++ Redistributable (' + GetExceptionMessage + '). ' +
              'Install it from https://aka.ms/vs/17/release/vc_redist.x64.exe and run this setup again.';
    Exit;
  end;
  Target := ExpandConstant('{tmp}\vc_redist.x64.exe');
  if not SignedByMicrosoft(Target) then
  begin
    Result := 'The downloaded runtime installer does not carry a valid Microsoft signature, so it was not run.';
    Exit;
  end;
  Started := Exec(Target, '/install /quiet /norestart', '', SW_HIDE, ewWaitUntilTerminated, Code);
  Log('Microsoft Visual C++ Redistributable installer: started=' + IntToStr(Integer(Started)) + ', exit code ' + IntToStr(Code));
  if not Started then
    Result := 'Could not start the Microsoft Visual C++ Redistributable installer.'
  else if (Code = 3010) or (Code = 1641) then
    NeedsRestart := True
  else if (Code <> 0) and (Code <> 1638) then   { 1638: a newer version is already installed }
    Result := 'The Microsoft Visual C++ Redistributable installer failed (exit code ' + IntToStr(Code) + ').';
end;

function PrepareToInstall(var NeedsRestart: Boolean): String;
var
  Problem, Answer: String;
begin
  Result := '';
  { /ForceVCRedist=1 exercises the download path even when a current runtime is present (used by CI). }
  if (not VCRuntimeCurrent) or (ExpandConstant('{param:ForceVCRedist|0}') = '1') then
  begin
    Problem := InstallVCRuntime(NeedsRestart);
    if Problem <> '' then
    begin
      if WizardSilent then
        Result := Problem
      else if MsgBox(Problem + #13#10#13#10 + 'Continue installing El-Modem anyway? It will not start until the ' +
                     'Microsoft Visual C++ Redistributable 2015-2022 (x64) is installed.', mbError, MB_YESNO) = IDNO then
        Result := Problem;
    end;
  end;
end;

function DriverPayloadExists: Boolean;
begin
  Result := FileExists(ExpandConstant('{app}\driver\wudf.cat'));
end;

{ The driver is test-signed, so installing it means trusting a certificate on
  this machine. Say exactly what that does before doing it, and let a silent
  install be told not to. }
procedure InitializeWizard();
var
  Response: Integer;
begin
  if (not DriverPayloadExists) or WizardSilent then
  begin
    if (not DriverPayloadExists) and WizardSilent then
      Log('No signed driver payload present; skipping driver installation.');
    Exit;
  end;
  Response := MsgBox(
    'The El-Modem virtual COM driver is included with this installer.' + #13#10#13#10 +
    'It is currently TEST-SIGNED, so installing it adds a certificate to this ' +
    'computer''s Trusted Root and Trusted Publishers stores. Anything signed with ' +
    'that certificate would then be trusted here without kernel code integrity review.' + #13#10#13#10 +
    'The certificate is removed again when El-Modem is uninstalled.' + #13#10#13#10 +
    'Install the driver now?', mbConfirmation, MB_YESNO);
  if Response = IDNO then
  begin
    WizardNoDriver := '1';
    Log('User declined driver installation; the driver and its certificate will be skipped.');
  end;
end;

procedure CurStepChanged(CurStep: TSetupStep);
var
  ResultCode: Integer;
  CommandLine, Output: String;
  Installed: Boolean;
begin
  if CurStep <> ssPostInstall then Exit;
  if not DriverPayloadExists then Exit;
  if WizardNoDriver = '1' then Exit;

  { Run after installation so the rest of El-Modem is already in place if this
    fails: a missing COM port does not make the program unusable. }
  CommandLine :=
    '-NoProfile -NonInteractive -ExecutionPolicy Bypass -File "' +
    ExpandConstant('{app}\driver\install_driver.ps1') +
    '" -DriverPackage "' + ExpandConstant('{app}\driver') + '"';
  Installed := Exec(ExpandConstant('{sys}\WindowsPowerShell\v1.0\powershell.exe'),
                    CommandLine, '', SW_HIDE, ewWaitUntilTerminated, ResultCode);

  { Exit 0 = installed, 2 = staged but no COM port (usually no WDK present). }
  if (not Installed) or (ResultCode = 1) then
    Log('Driver installation failed (exit ' + IntToStr(ResultCode) + '); El-Modem installed without it.')
  else if ResultCode = 2 then
    Log('Driver package staged but no COM port was created (exit 2). ' +
        'Install the Windows Driver Kit and run: devcon.exe install virtualserial2um.inf UMDF\VirtualSerial2um')
  else
    Log('El-Modem virtual COM driver installed.');
end;

{ Removing the test certificate is the point of uninstalling: while it is
  trusted, this machine accepts anything signed with it. Runs before Inno
  deletes {app}, because the script lives there. }
procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
var
  ResultCode: Integer;
  Installed: Boolean;
begin
  if CurUninstallStep <> usUninstall then Exit;
  if not FileExists(ExpandConstant('{app}\driver\uninstall_driver.ps1')) then Exit;
  if WizardNoDriver = '1' then Exit;
  Installed := Exec(ExpandConstant('{sys}\WindowsPowerShell\v1.0\powershell.exe'),
                    '-NoProfile -NonInteractive -ExecutionPolicy Bypass -File "' +
                    ExpandConstant('{app}\driver\uninstall_driver.ps1') +
                    '" -DriverPackage "' + ExpandConstant('{app}\driver') + '"',
                    '', SW_HIDE, ewWaitUntilTerminated, ResultCode);
  if Installed then
    Log('Driver and test certificate removal finished (exit ' + IntToStr(ResultCode) + ').')
  else
    Log('Driver removal did not complete (exit ' + IntToStr(ResultCode) +
        '). Run: powershell -File "' + ExpandConstant('{app}\driver\uninstall_driver.ps1') + '"');
end;
