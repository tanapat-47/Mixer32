Mixer32 Studio Practice — M32/X32-inspired UI build

This is a Windows practice mixer prototype with a dark, metallic M32/X32-inspired interface.
It is NOT official Behringer software and is not a 1:1 recreation of the hardware mixer.

Features in this prototype:
- 32 audio channel strips
- Load multitrack audio files or a folder of tracks (WAV, FLAC, AIFF, OGG)
- Fader, pan, mute, solo, polarity, HPF, gate, compressor, 4-band EQ controls
- 16 bus masters, 8 simplified FX slots, DCA and mute groups
- Save/load scenes and project settings as JSON/.m32p
- Play/stop and export a stereo WAV mix

Run on Windows:
1. Install Python 3.11 or newer from https://www.python.org/downloads/ (enable Add Python to PATH).
2. Extract this ZIP fully to a normal folder.
3. Double-click run_windows.bat and wait for dependency installation to finish.
4. The app window should open. If it doesn't, open CMD in this folder and run:
   .venv\Scripts\python.exe app.py

Build an EXE:
- Double-click build_exe_windows.bat
- When complete, find the app in dist\Mixer32StudioPractice\Mixer32StudioPractice.exe

Notes:
- First launch downloads Python packages and needs an internet connection.
- This is a learning/practice prototype; some routing and effects are simplified approximations, not identical to a real M32/X32 DSP engine.
- Projects store source audio paths, so keep source files in the same locations when reopening a project.
