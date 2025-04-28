# -*- coding: utf-8 -*-


import re

import os
import click
import json

import logging

# commandlineの引数でsrtファイル名を取得して正規表現でセリフ部分のみを抜き出して文字列のリストを作って出力する。
@click.command()
@click.argument('srt_file', type=click.Path(exists=True))
@click.option('--output', '-o', type=click.Path(), help='Output file name')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
def extract_dialogue(srt_file, output, verbose):
    """
    Extract dialogue lines from an SRT file and output as a list of strings.
    """
    try:
        with open(srt_file, 'r', encoding='utf-8') as file:
            content = file.read()

        # Regular expression to match dialogue lines in SRT files
        #dialogue_lines = re.findall(r'^\d+\n(?:\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}\n)?(.*(?:\n(?!\d+\n|\d{2}:\d{2}:\d{2},\d{3}).*)*)', content, re.MULTILINE)
        dialogue_lines = re.findall(r'\d+\n(?:\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}\n)?(.*(?:\n(?!\d+\n|\d{2}:\d{2}:\d{2},\d{3}).*)*)', content, re.MULTILINE)
        for i in range(10):
            print(i, dialogue_lines[i])
        # Clean up and flatten the list of dialogues
        dialogues = [{'line':[line.strip()]} for group in dialogue_lines for line in group.split('\n') if line.strip()]

        # Output the result as a JSON list
        if output:
            with open(output, 'w', encoding='utf-8') as outfile:
                json.dump(dialogues, outfile, ensure_ascii=False, indent=2)
        else:
            print(json.dumps(dialogues, ensure_ascii=False, indent=2))

    except Exception as e:
        logging.error(f"An error occurred: {e}")

if __name__ == '__main__':
    extract_dialogue()
