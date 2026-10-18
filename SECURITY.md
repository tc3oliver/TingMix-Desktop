# Security

Please don't report security problems in public issues. Use GitHub's private vulnerability reporting instead:
https://github.com/tc3oliver/TingMix-Desktop/security/advisories/new

Tell me what you found, how to reproduce it, and which commit or version it affects. I'll reply within a week. Please give
me time to fix it before you publish anything, and say if you'd like to be credited.

Only the latest commit on `main` is supported for now. There are no releases yet.

Things I especially want to hear about:

- Getting a model installed that shouldn't be: a bad signature, a wrong size or hash, path tricks in a zip or folder
  import, a non-data file, or an older package replacing a newer one.
- Anything that sends audio, text or identifiers off the machine, or opens a listening port.
- The web view or a crafted file reaching something it shouldn't, such as the file system, a shell or an arbitrary URL.
- Crashes or memory bugs when reading untrusted audio, model or language-model files.
- The updater accepting an unsigned or modified update.

Bugs in upstream projects that this app can't trigger should go upstream. So should anything that needs a modified build
or an already compromised machine.

No private keys are stored in this repository. If you find a credential anywhere in it, including its history, please
report it the same way.
