# Personal ChatGPT Knowledge Center – Structured AI Conversation Database & Clustering Pipeline

## Purpose
**Inspiration**:
As part of my personal development in AI, I wanted a way to capture my interactions with Artifical Intelligence. This project transforms my entire ChatGPT conversation history into a structured, searchable, and cluster-based knowledge center. It leverages natural language embeddings, unsupervised clustering, and database design to make my AI dialogue truly mine.

## Description
The project provides a robust pipeline for:

**Conversation Parsing**: Converts conversations.json from the ChatGPT export into structured message rows.

**Embedding & Clustering**: Uses sentence embeddings and HDBSCAN to group similar conversation threads into topics.

**SQLite Knowledge Base**: Stores clustered messages and topics in a relational database for future querying, filtering, or UI development.

## Architecture & Design
The flow is described below:

Conversation Parsing (structure_json.py):

Parses message trees and flattens content from conversations.json

Converts each message into (conversation title, role, message text, timestamp)

Embedding & Topic Modeling:

Embeds messages using sentence-transformers (MiniLM-L6-v2)

Clusters them using HDBSCAN to identify high-level topic groups

Database Construction (knowledge.db):

Creates topic and message tables

Saves all data with references between clustered messages and their topics

## Data Flow
**Input**:
conversations.json exported from ChatGPT

**Processing**:
Flattening nested messages
Text cleaning and extraction
Sentence embedding
Unsupervised topic clustering

**Output**:

knowledge.db SQLite file containing:

topic(id, title, summary)

message(id, topic_id, conv, role, ts, text)

## Usage
1. Export your ChatGPT Data
Download your conversations.json from https://chat.openai.com/export
Place it in the project root directory.
