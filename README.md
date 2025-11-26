# kirkify.py

Groundbreaking engine behind the hit website, kirkify.me, which some are saying is the most innovative software since Facebook or Google.

It's a relatively simple python script to kirkify any image or video given.

## Setup

It's a bit of a pain to setup but I promise it's worth it.

First download the face swapping AI (528mb)

```bash
curl -o "inswapper_128.onnx" https://bk4vz20t6s.ufs.sh/f/5eVwDsd8R3jL5kumGF8R3jLVwUJfdOu8cQ4ymMqAFeW7zrEX
```

Next, install the requirements.

```bash
pip install -r requirements.txt
```

After, initialize the script (you don't technically have to but it saves time on first run)

```bash
python3 kirkify.py init
```

Finally, if you don't already have it installed, the video kirkifier requires `ffmpeg` to be installed

```bash
apt install ffmpeg
```

## Usage

```bash
python3 kirkify.py <input_media> [output_path]
```
