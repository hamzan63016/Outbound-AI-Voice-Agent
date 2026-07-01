**# To test the program use your own API key, email and security password.**

# Outbound-AI-Voice-Agent
This project is an automated, outbound AI Voice Agent built specifically for automated debt collection and customer account audits. It operates as an interactive voice response (IVR) state machine that speaks to customers, listens to their responses, and updates account statuses accordingly.

**Here is how the core modules function together:**

Database & Dashboard Tracking: A local SQLite backend tracks customer account profiles (balances, due dates, contact validation). It launches an integrated Flask web server (http://127.0.0.1:5000) providing a live visual dashboard for monitoring operational statuses.

Dual-Engine Speech Synthesis (TTS): The agent vocalizes statements utilizing a high-quality cloned voice sample (reference_voice.wav) via the Coqui XTTS engine. If dependencies drop or fail, it seamlessly uses Microsoft's cloud-based edge-tts infrastructure as a fallback to prevent conversational freezes.

Acoustic Voice Activity Detection (VAD): The system constantly tracks incoming microphone feeds via a background thread listener. It monitors decibel ambient thresholds, automatically tuning out room noise to accurately pinpoint exactly when a customer starts and stops speaking.

Hybrid Intent Parsing Layer: Customer speech transcripts are parsed through an optimization gate. Simple structural verification states (identity approval/denial) are checked locally via fast hardcoded keyword arrays. Complex transactional conversations (payment negotiations, partial installment setups) automatically route to an underlying gpt-4o-mini language model for contextual parsing.

Session Auditing & Record Merging: Throughout the call, the program records both audio streams. When a hangup occurs, it sequences the agent's generated speech files and the customer's recorded responses into a chronological master audio file (recordings/call_session_...wav) while simultaneously auto-generating an AI executive audit summary.

**Functionality Flowchart:**

<img width="217" height="369" alt="Screenshot 2026-06-29 151826" src="https://github.com/user-attachments/assets/4594a18f-8b06-4e5e-9dda-b54126499598" />

**Dashboard:**

<img width="1920" height="1080" alt="Screenshot (11)" src="https://github.com/user-attachments/assets/ab4555e7-43b7-4f92-a8f2-94ff08dd3c43" />

**E-Mail:**
<img width="1920" height="1080" alt="Screenshot (12)" src="https://github.com/user-attachments/assets/4cc29e02-0b87-44a8-8105-56380847f708" />

**Call Recording:**
 [call_session_1_20260629_154416.wav](https://github.com/user-attachments/files/29547615/call_session_1_20260629_154416.wav)
