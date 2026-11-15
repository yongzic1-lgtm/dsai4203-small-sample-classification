# Recording guide: the team video

**Status:** written 2026-11-01 by Claude Code (Anthropic, Claude Opus 5.5; workflow agent `video_kit`). Nothing has been recorded or installed. The commands in §2 were tested on the team laptop with TeX Live 2024 (in a scratch folder instead of `D:\video_raw\slides_build`); the menu names of Zoom, Teams, PowerPoint, Clipchamp, OBS and HandBrake come from their current versions and may differ slightly on your machine.

Spoken text: `video/SCRIPT.md`. Slides: `video/slides.pdf` (source `video/slides.tex`). Limits and storyboard: `video/PLAN.md`. Schedule: record 11-24 to 11-26, package check 11-27 (`docs/TIMELINE.md`).

## 0. What the video must meet

- **Length:** at most 5:00, whatever the team size. Aim for 4:40.
- **People:** every member presents, and each one's face is on camera while they speak.
- **Size:** at most 300 MB. `scripts/package_deliverables.py` checks 300,000,000 bytes (the stricter, decimal reading).
- **Language:** English (whether the course requires it is still a TODO in `PLAN.md`; check Canvas).
- **One file:** exactly one `.mp4`, `.mov` or `.mkv` anywhere under `video/` (subfolders count), named `video/<TEAM_NAME>.mp4`. Keep raw recordings outside the project, for example in `D:\video_raw\`.
- **Stays inside the team:** the video shows our results. Do not upload it or drafts of it to YouTube, Bilibili or any public link before the deadline (competition rule 5); share drafts only in the team's private folder. It reaches the course only inside the team zip on Canvas.

## 1. Prepare (once, all members, about one hour)

1. **Extensions.** Agree who owns which extension (`project.yaml`, `reports/contribution_table.md`). Only owned extensions get a block and a slide.
2. **Slides.** Open `video/slides.tex` in a text editor and fill the block marked `EDIT HERE`: team name, Kaggle team name, member names, extension owners, `\Eonefalse` (etc.) for every extension nobody owns, and the takeaway sentences. Save as UTF-8 (Notepad does this by default). Rebuild the PDF (§2). If you cannot rebuild, present the current `slides.pdf` and skip the slides of unowned extensions.
3. **Script.** Each speaker rewrites their lines in `SCRIPT.md` in their own words, reads them aloud with a timer, and checks that they can explain every number (sources: `SCRIPT.md` §6). Say "I" only for work you did yourself.
4. **Rehearsal.** One full run with the slides and a stopwatch. If it runs over 4:45, use the cut list at the top of `SCRIPT.md`.

## 2. Rebuild the slides (after every change to `slides.tex`)

XeLaTeX can fail on the non-ASCII project path, so build in an ASCII folder and copy the PDF back (PowerShell; TeX Live and the Calibri font are already on the team laptop):

```powershell
$root = "D:\polyu_class\新建文件夹"           # project root
$b = "D:\video_raw\slides_build"             # any folder whose path is plain ASCII
New-Item -ItemType Directory -Force "$b\figures" | Out-Null
Copy-Item -LiteralPath "$root\video\slides.tex" -Destination $b
Copy-Item -Path "$root\reports\figures\*.png" -Destination "$b\figures"
Push-Location $b
xelatex -interaction=nonstopmode slides.tex
xelatex -interaction=nonstopmode slides.tex      # second run fixes the slide numbers
Pop-Location
Copy-Item -LiteralPath "$b\slides.pdf" -Destination "$root\video\slides.pdf"
```

Then open `video/slides.pdf` and check: 11–14 pages (10 fixed slides plus one per owned extension), names right, no text in the bottom-right corner. That corner (3.0 × 1.7 in, about 430 × 245 px in a 1080p recording) is kept empty on every slide for the camera picture; set `\camguidetrue` in `slides.tex` to see it outlined while rehearsing, and back to `\camguidefalse` for the final PDF.

Slide images for PowerPoint (route B), in the same folder:

```powershell
Push-Location "D:\video_raw\slides_build"; pdftoppm -png -r 144 slides.pdf slide; Pop-Location
```

This writes `slide-01.png`, `slide-02.png`, … at 1920 × 1080.

## 3. Set up each computer before recording

- **Screen.** Close everything you do not need: other browser tabs, email, chat, Canvas, file windows that show personal paths. **Never show** a Kaggle page (Account or Settings, the API token, the leaderboard), `kaggle.json`, Colab, or anything with a password or token. Show only the slides.
- **Notifications off.** Windows 11: Settings → System → Notifications → turn on *Do not disturb*. Quit chat apps.
- **Slides full screen.** Open `slides.pdf` in Adobe Acrobat Reader (View → Full Screen Mode, `Ctrl+L`) or in another PDF viewer that shows one whole page per screen; move with the arrow keys. A display at 1920 × 1080 (or 1280 × 720) and 100 % scaling makes the slide fill the screen.
- **Camera.** At eye level, light in front of you (a window behind you darkens your face), plain background, your face about a third of the camera frame.
- **Sound.** Headset or earbud microphone, quiet room, same distance to the microphone throughout. Record 10 s and play it back before the real take. If several members sit in one room, only one microphone is on.

## 4. Record: choose one route

| Route | Best for | Faces on screen | Needs |
|---|---|---|---|
| A. Zoom or Teams meeting | members in different places, one session | the speaker's camera next to the slides; gallery view for opening and closing | Zoom desktop app or a Teams account that may record |
| B. PowerPoint *Record* with camera | one computer, or each member records their own slides | camera picture in the bottom-right corner of each slide | PowerPoint (Microsoft 365) and the slide images of §2 |
| C. Clipchamp screen and camera | each member records alone, then one member joins the clips | camera picture over the slides | nothing new: Clipchamp comes with Windows 11 |
| D. OBS Studio | most control over layout and quality | webcam placed in the bottom-right corner | installing OBS (optional; the member decides) |

### Route A: Zoom (or Teams)

1. **Settings (host, Zoom desktop app):** Settings → Recording → *Store my recordings at* `D:\video_raw`; tick *Record video during screen sharing* and *Place video next to the shared screen in the recording*; *Optimize for 3rd party video editor* helps later editing.
2. **Join:** everyone with camera on and headset; only the current speaker unmuted.
3. **Opening:** nobody shares; switch to *Gallery view*; Record → *Record on this computer*. A, B and C say their opening lines.
4. **Slides:** one presenter shares only the PDF window (Share Screen → that window, not the whole desktop) in full-screen mode and moves to the next slide at the points marked *(slide N)* in the script. Each speaker talks while their slides are up; their camera picture appears next to the slides.
5. **Closing:** stop sharing, *Gallery view*, everyone says their takeaway.
6. **Stop** recording. Zoom converts the file when the meeting ends; the MP4 lands in the folder from step 1.
7. **Teams instead:** More (…) → *Record and transcribe* → *Start recording*. The recording goes to the organiser's OneDrive (or the meeting chat), where it can be downloaded as MP4. Check in a short test that each speaker's video is visible while their slides are shared; if not, use route B or C.

### Route B: PowerPoint *Record* with camera

1. Make the slide images (§2). PowerPoint: Insert → *Photo Album* → *New Photo Album* → *File/Disk* → select all `slide-*.png` → Picture layout *Fit to slide* → *Create*. Delete the album's own title slide. Check Design → Slide Size → *Widescreen (16:9)*.
2. Record → *From Beginning*. Turn on camera and microphone; the camera picture sits in the bottom-right corner, which the slides keep empty. Speak slide by slide; a slide can be re-recorded on its own.
3. Several members: take turns at one computer, or each records their slides in a copy of the file; copying recorded slides into one file keeps their recordings. For the opening and closing, sit together in front of one camera, or record those two parts in route C and join them in §5.
4. File → Export → *Create a Video* → *Full HD (1080p)* → *Use Recorded Timings and Narrations* → *Create Video*, saved as MP4 in `D:\video_raw`.

### Route C: Clipchamp (no installation)

1. Start menu → Clipchamp → *Create a new video* → *Record & create* → *Screen and camera*.
2. Choose the screen that shows the full-screen PDF, your camera and your microphone; record your segment. One recording can be up to 30 minutes.
3. Clipchamp places the camera picture over the screen recording; in the editor, move it to the bottom-right corner and resize it so that it covers no text.
4. Each member records their own segments and puts the files in the team's private folder; one member joins them (§5).

### Route D: OBS Studio (optional install)

1. Settings → Video: base and output resolution 1920 × 1080, 30 fps. Settings → Output → Recording: path `D:\video_raw`, format MKV (survives a crash) or MP4, a hardware encoder if offered (NVENC, QuickSync, AMF) or x264; quality CRF/CQP about 20–23. Audio bitrate 160 kbps.
2. Scene sources: *Display Capture* (or *Window Capture* of the PDF viewer) and *Video Capture Device* (the webcam). Resize the webcam to about 430 × 245 px and move it into the bottom-right corner. Keep the microphone on and mute *Desktop Audio*.
3. *Start Recording* / *Stop Recording* for each segment. MKV files: File → *Remux Recordings* → MP4.

## 5. Join and trim (Clipchamp, or ffmpeg if installed)

1. Clipchamp → *Create a new video* → import all clips → drag them onto the timeline in script order.
2. Cut the silence at the start and end of each clip (split tool, or drag the clip edges).
3. Opening and closing: if members recorded apart, place their camera clips side by side (resize each to a third of the frame) or one after another, so every face is seen.
4. The total length is shown under the timeline: at most 4:55.
5. *Export* → *1080p* (720p if the file comes out above 300 MB). Clipchamp saves to Downloads; move and rename the file to `video\<TEAM_NAME>.mp4` (§7).

With ffmpeg, clips with identical encoding settings can be joined without re-encoding (from `PLAN.md`): write `D:\video_raw\list.txt` with one line `file 'part1.mp4'` per clip, then

```powershell
ffmpeg -f concat -safe 0 -i D:\video_raw\list.txt -c copy D:\video_raw\final_edit.mp4
```

(Clips with different settings: leave out `-c copy` so that ffmpeg re-encodes.)

## 6. Make it smaller (only if it is above 300 MB)

300 MB over 300 s allows about 8 Mbit/s. A 1080p slide video with a camera picture usually needs 2–4 Mbit/s, about 75–150 MB for five minutes, so most exports fit directly.

- **Clipchamp:** export again at 720p.
- **HandBrake** (free and open source; installing it is the member's choice): *Open Source* → the MP4; Preset *General → Fast 1080p30* (or *Fast 720p30*); Summary: Format *MP4*, tick *Web Optimized*; Video: encoder *H.264 (x264)*, framerate 30, *Constant Quality* RF 24 (26–28 if still too large); Audio: AAC, 160 kbps; *Save As* `video\<TEAM_NAME>.mp4` → *Start Encode*.
- **ffmpeg** (not installed on the team laptop on 2026-11-01; for example `winget install --id Gyan.FFmpeg -e`, the member's decision). Command from `PLAN.md`: 720p, 30 fps, H.264 + AAC:

```powershell
ffmpeg -i D:\video_raw\final_edit.mp4 -vf "scale=-2:720,fps=30" -c:v libx264 -preset slow -crf 26 -c:a aac -b:a 128k -movflags +faststart video\TODO_TEAM_NAME.mp4
```

Still too large: raise `-crf` to 28–30 (a larger number gives a smaller file). Check length and size with ffmpeg installed:

```powershell
ffprobe -v error -show_entries format=duration,size -of default=noprint_wrappers=1 video\TODO_TEAM_NAME.mp4
```

## 7. Check the final file

- **Name:** `video\<TEAM_NAME>.mp4`, with `<TEAM_NAME>` = `team.name` in `project.yaml`, using only letters, digits, spaces, `.`, `_` and `-`. The zip gets the same name (`package_deliverables.py` turns other characters into `_`).
- **Only one video file** under `video\`, subfolders included; raw clips stay in `D:\video_raw\`.
- **Size:** File Explorer → right-click → Properties → *Size*: the number of bytes in brackets must be at most 300,000,000. Explorer's "MB" are binary megabytes, so the limit shows there as 286 MB. In PowerShell: `(Get-Item -LiteralPath "video\<TEAM_NAME>.mp4").Length`.
- **Length:** Properties → Details → *Length* at most 00:05:00 (aim for 00:04:40; never above 00:04:55 to be safe).
- **Watch it once, end to end,** ideally a member who did not edit it: every member's face visible while they speak; voices clear and at a similar volume; slides readable and in step with the words; no notification, browser tab, file path, e-mail or Kaggle page anywhere.
- **Package check:** `.venv\Scripts\python.exe scripts\package_deliverables.py` (dry run). The `[video]` line must say OK and name this file.
- **Not in git:** `.gitignore` already excludes `video/**/*.mp4|mov|mkv`; do not force-add it.

## 8. Checklist (copy into the team chat)

- [ ] Extension owners decided; `slides.tex` names, owners, extension switches and takeaways filled; `slides.pdf` rebuilt
- [ ] Each speaker rewrote their lines and can explain every number (`SCRIPT.md` §6)
- [ ] Full rehearsal at most 4:45
- [ ] Notifications off; only the slides on screen; no Kaggle Account, Settings or API-token page, no `kaggle.json`, no e-mail or chat
- [ ] Camera at eye level, face lit from the front and visible whenever that member speaks
- [ ] Headset microphone, quiet room, 10-second sound test played back
- [ ] Opening and closing show every member's face
- [ ] Final length at most 5:00 (aim 4:40)
- [ ] File at most 300,000,000 bytes; MP4 with H.264 video and AAC audio
- [ ] Named `video\<TEAM_NAME>.mp4`; the only video file under `video\`
- [ ] Watched once in full by a second member
- [ ] `package_deliverables.py` dry run: `[video]` OK
- [ ] Not uploaded to any public site; not committed to git
