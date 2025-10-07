1.  Change your VSCode setting to Unity frieldly stuff
    ```~/library/Applicatio Support/Code/User/setting.json
    {
        "editor.fontSize": 14,
        "editor.autoClosingBrackets": "never",
        "editor.autoClosingComments": "never",
        "editor.autoClosingQuotes": "never",
        "editor.bracketPairColorization.independentColorPoolPerBracketType": true,
        "terminal.integrated.inheritEnv": false,
        "notebook.output.wordWrap": true,
        "jupyter.widgetScriptSources": [
            "jsdelivr.com",
            "unpkg.com"
        ],
        "git.autofetch": true,
        "diffEditor.maxComputationTime": 0,
        "git.ignoreLegacyWarning": true,
        "jupyter.askForKernelRestart": false,
        "workbench.colorTheme": "Solarized Dark",
        "workbench.preferredDarkColorTheme": "Catppuccin Mocha",
        "window.autoDetectColorScheme": true,
        "workbench.colorCustomizations": {
            "notebook.findMatchBackground": "#ffb900",
            "notebook.findMatchHighlightBackground": "#ffdd55",
            "notebook.findMatchBorder": "#ff0000",
            "editor.findMatchBackground": "#ffb900",
            "editor.findMatchHighlightBackground": "#ffdd5580",
            "editor.findMatchBorder": "#ff0000",
            "editor.findRangeHighlightBackground": "#ffdd5580"
        },
        "github.copilot.nextEditSuggestions.enabled": true,
        "geminicodeassist.inlineSuggestions.enableAuto": true,
        "diffEditor.codeLens": true,
        "python.createEnvironment.trigger": "off",
        "workbench.editorAssociations": {
            "*.copilotmd": "vscode.markdown.preview.editor",
            "*.hdf5": "vscode-hdf5-viewer.preview"
        },
        "remote.SSH.useLocalServer": true,
        "remote.SSH.connectTimeout": 30,
        "remote.SSH.defaultForwardedPorts": [

        ],
        "remote.SSH.remotePlatform": {
            "192.168.12.77": "linux"
        },
        "workbench.iconTheme": "catppuccin-mocha",
        "git.confirmSync": false,
        "geminicodeassist.project": "cosmic-carving-pg250",
        "remote.extensionKind": {
        

            "pub.name": [
                "ui"
            ]
        },
        "remote.SSH.localServerDownload": "always",
        "remote.SSH.lockfilesInTmp": true,
        "remote.SSH.preconnect": "",
        "remote.SSH.serverInstallPath": {
            "asc-*": "/home/malpas.1/.vscode-server"
        }
    }
    ```
    Don't ask me which if these are important. Maybe this one `"remote.SSH.localServerDownload": "always"`, this one `"remote.SSH.lockfilesInTmp": true`, this one `remote.SSH.serverInstallPath": {"asc-*": "/home/malpas.1/.vscode-server"}`, and this one `"remote.SSH.useLocalServer": true`.

2.  Open an ondemand terminal or ssh in to unity

3.  Create a job
    ```
    (base) [<user>@<login-node> <dir>]$ srun --pty -n 4 --mem=16G bash
    (base) [<user>@<node> <dir>]$
    ```

4.  Edit you `~/.ssh/config` (malpas.1 should be replaced with your user)
    ```
    Host asc-jump
      HostName jump.asc.ohio-state.edu
      Port 2200
      User malpas.1
      IdentityFile ~/.ssh/id_rsa

    Host asc-login
      HostName unity.asc.ohio-state.edu
      User malpas.1
      ProxyJump asc-jump
      IdentityFile ~/.ssh/id_rsa

    # <— replace u089 each session with the <node> you actually got
    Host asc-u089
      HostName u089
      User malpas.1
      ProxyJump asc-login
      IdentityFile ~/.ssh/id_rsa
    ```

5.  ssh into the node:
    ```
    ssh asc-089
    # accept the DUO push
    yes # when prompted
    ```

6.  Click the >/< button at the bottom left of your VSCode window.
    `Connect to Host`
    `asc-u089`
    *accept the DUO push*
    *open a folder* (don't open two folder's in a workspace; it doesn't like it)
