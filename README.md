# Meeting Notes
## Use AI to Take Detailed Meeting Notes
Translate and summarize meeting notes, using openAI.  This program runs locally on your computer.  It uses the openAI API to translate speech to text, and then uses the openAI API to summarize the text.  It then saves the results to a file.

## Example Command Line Usage

The general format for calling from the command line is:

`python3 note_taker.py -f "/path/to/file.m4a"`

For example:

`python3 note_taker.py -f "/private/var/folders/jy/fqc1n18x00n2r4qhmp94lr3w0000gn/T/com.apple.VoiceMemos/.com.apple.uikit.itemprovider.temporary.SW8bHY/2023.05.08 - Synthea Handoff.m4a"`

## Features

This program summarizes the transcript of your meeting and provide some light analysis of the meeting.

1. Make a list of action items and due outs.
2. Make a list of decisions that were made.
3. Make a list of important topics discussed.
4. Make a list of questions that were asked and answered.
5. Make a list of questions that were asked and not answered.
6. Make a list of keywords for search.
7. Bullet points on the meeting.  

# Development Roadmap
These are things to do soon!
1. (Add Recording Function._)[https://realpython.com/playing-and-recording-sound-python/#recording-audio] . . . can't seem to figure out how to get permalinks to Apple Voice Memos.
1. Make an install script; installs the files, makes the directories.
5. Analyze for mood and sentiment of the meeting.  Justify.
7. Add in keywords to the final docx.

# Instalation Notes

0. Install portaudio `brew install portaudio`
1. Setup a virtual environment.  Install the requirements.txt file.
2. Copy meeting_notes_config.key.example to meeting_notes_config.key
2. Update the four variables within the meeting_notes_config file.
3. Add your virtual environment folder into the .gitignore file so it doesn't sync to github.
4. Make a directory "gpt_logs"
5. Copy over "NoteTaker.command.example" to "NoteTaker.command"
6. Copy over "NoteTaker.command.example" to "NoteTaker.command"
7. Make executable with a chmod +x 

## Virtual Environment Notes

### Use the Venv
This is a note to myself on how to set up a virtual environment for meeting_notes
```
deactivate
```
Activate it
```
source openai202305/bin/activate
```

### Make the VENV

    '''    
    python3 -m venv openai202305
    source openai202305/bin/activate
    pip install --upgrade pip 
    pip install -r requirements.txt
    # Check that everything installed
    pip list
    '''