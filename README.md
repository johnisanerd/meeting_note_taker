# Meeting Notes
## Use AI to Take Detailed Meeting Notes
Translate and summarize meeting notes, using openAI.  This program runs locally on your computer.  It uses the openAI API to translate speech to text, and then uses the openAI API to summarize the text.  It then saves the results to a file.

## Summarize the meeting, provide:
1. Make a list of action items and due outs.
2. Make a list of decisions that were made.
3. Make a list of important topics discussed.
4. Make a list of questions that were asked and answered.
5. Make a list of questions that were asked and not answered.
6. Make a list of keywords for search.

# Development Roadmap

1. Make this right-click deployable.
2. Make an install script. 
3. Consolidate functions, cleanup.
4. Documentation
5. Add proper logging.
6. Analyze for mood and sentiment of the meeting.  Justify.
7. Add in 'click' and right click capabilities.
8. Add in keywords to the final docx.

# Instalation Notes

1. Setup a virtual environment.  Install the requirements.txt file.
2. Copy meeting_notes_config.key.example to meeting_notes_config.key
2. Update the three variables within the meeting_notes_config file.
3. Add your virtual environment folder into the .gitignore file so it doesn't sync to github.  

## Virtual Environment Notes

### Use the Venv
This is a note to myself on how to set up a virtual environment for meeting_notes
```
deactivate
```
Activate it
```
source /Users/johncole/Github/meeting_notes/openai202305/bin/activate
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