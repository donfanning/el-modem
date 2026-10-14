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

[Icons]
; El-Modem is a terminal application: the shortcut opens it in a console window.
Name: "{group}\El-Modem"; Filename: "{app}\el-modem.exe"; WorkingDir: "{userdocs}"
Name: "{group}\Uninstall El-Modem"; Filename: "{uninstallexe}"

[Code]
const
  VCRedistUrl = 'https://aka.ms/vs/17/release/vc_redist.x64.exe';
  VCRegistryKey = 'SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64';
  MinimumMinor = 40;   { 14.40 = Visual Studio 2022 17.10; the runtime the frozen build was checked with }

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
  Script: String;
begin
  { The download URL is a moving target, so a pinned hash is impossible. Require a valid
    Authenticode signature whose signer is Microsoft Corporation instead. }
  Script := '$s = Get-AuthenticodeSignature -LiteralPath ''' + FileName + '''; ' +
            'if ($s.Status -eq ''Valid'' -and $s.SignerCertificate.Subject -match ''O=Microsoft Corporation'') { exit 0 } else { exit 1 }';
  Result := Exec(ExpandConstant('{sys}\WindowsPowerShell\v1.0\powershell.exe'),
                 '-NoProfile -NonInteractive -ExecutionPolicy Bypass -Command "' + Script + '"',
                 '', SW_HIDE, ewWaitUntilTerminated, Code) and (Code = 0);
end;

function InstallVCRuntime(var NeedsRestart: Boolean): String;
var
  Target: String;
  Code: Integer;
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
  if not Exec(Target, '/install /quiet /norestart', '', SW_HIDE, ewWaitUntilTerminated, Code) then
    Result := 'Could not start the Microsoft Visual C++ Redistributable installer.'
  else if (Code = 3010) or (Code = 1641) then
    NeedsRestart := True
  else if (Code <> 0) and (Code <> 1638) then   { 1638: a newer version is already installed }
    Result := 'The Microsoft Visual C++ Redistributable installer failed (exit code ' + IntToStr(Code) + ').';
end;

function PrepareToInstall(var NeedsRestart: Boolean): String;
var
  Problem: String;
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
