"""
WhatsApp <-> Claude bot for a Chennai Airbnb property management service.

Flow:
  Guest sends WhatsApp message
    -> Twilio forwards it to this FastAPI webhook (POST /whatsapp)
    -> We send the message + system prompt to Claude
    -> Claude's reply is sent back to the guest via Twilio
    -> If the reply is tagged [ESCALATE], the manager also gets a WhatsApp alert
"""

import os
from fastapi import FastAPI, Form, Request
from fastapi.responses import PlainTextResponse
from twilio.twiml.messaging_response import MessagingResponse
from twilio.rest import Client as TwilioClient
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()
client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

# ---------------------------------------------------------------------------
# Manager notification setup — sends YOU a WhatsApp message whenever the
# bot escalates a guest conversation. Requires these values in your .env:
#   TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN  (from Twilio Console > Account)
#   TWILIO_WHATSAPP_FROM   (your Twilio/sandbox WhatsApp number, e.g.
#                           "whatsapp:+14155238886")
#   MANAGER_WHATSAPP_NUMBER (your personal number, e.g. "whatsapp:+9198...")
# ---------------------------------------------------------------------------
TWILIO_ACCOUNT_SID = os.environ.get("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.environ.get("TWILIO_AUTH_TOKEN")
TWILIO_WHATSAPP_FROM = os.environ.get("TWILIO_WHATSAPP_FROM")
MANAGER_WHATSAPP_NUMBER = os.environ.get("MANAGER_WHATSAPP_NUMBER")

twilio_client = None
if TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN:
    twilio_client = TwilioClient(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)


def notify_manager(guest_number: str, guest_message: str, guest_reply: str) -> None:
    """Send the property manager a WhatsApp alert about an escalated chat."""
    if not (twilio_client and TWILIO_WHATSAPP_FROM and MANAGER_WHATSAPP_NUMBER):
        print(f"[ESCALATION - notify skipped, Twilio not configured] "
              f"{guest_number}: {guest_message}")
        return

    alert_text = (
        f"🚨 Escalation alert\n"
        f"Guest: {guest_number}\n"
        f"Message: {guest_message}\n"
        f"Bot told guest: {guest_reply}"
    )
    try:
        twilio_client.messages.create(
            from_=TWILIO_WHATSAPP_FROM,
            to=MANAGER_WHATSAPP_NUMBER,
            body=alert_text,
        )
    except Exception as e:
        print(f"[ERROR] Failed to notify manager: {e}")


# ---------------------------------------------------------------------------
# 1. Business knowledge Claude needs. Edit this for your properties.
#    Keep it factual and specific — this is what stops Claude from guessing.
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """
You are the WhatsApp assistant for CRAByo, an Airbnb property
management service in Chennai, India. You help guests and prospective
property owners.

PROPERTIES:
- List each property here with: name, area, check-in/out times, WiFi
  details, house rules, nearest landmarks.
  Chennai Properties:
  - "Ceyon Haven" — Check-in 3 PM, check-out 11 AM.
    Address: https://www.google.com/maps/place/Ayya+Aathiguru,+45,+5th+Main+Rd,+Venus+Colony+Extension,+Anna+Nagar+Extension,+Velachery,+Chennai,+Tamil+Nadu+600042,+India/@12.9725802,80.226679,15z/data=!4m16!1m9!3m8!1s0x3a525d9bf8a6324b:0x195148f3d6289636!2sAyya+Aathiguru,+45,+5th+Main+Rd,+Venus+Colony+Extension,+Anna+Nagar+Extension,+Velachery,+Chennai,+Tamil+Nadu+600042,+India!3b1!8m2!3d12.9726677!4d80.2267038!10e5!16s%2Fg%2F1q6jclr8b!3m5!1s0x3a525d9bf8a6324b:0x195148f3d6289636!8m2!3d12.9726677!4d80.2267038!16s%2Fg%2F1q6jclr8b?entry=ttu&g_ep=EgoyMDI2MDkwMS4wIKXMDSoASAFQAw%3D%3D
    WiFi: network "AaruB".
    WiFi Password: "Ceyon2023"
    Floor No: "1"
    Door No: "3"
    Door Code: "Abc123"
    Manager Name: "Swaminathan"
    No smoking indoors. 5 min walk from Velachery and Perungudi Railyway station. 5 Mins walk to Velachery Main Road.

GENERAL POLICIES:
- Standard check-in: 3 PM. Standard check-out: 11 AM.
- Late checkout possible if next guest isn't arriving same day — confirm
  availability, don't promise it outright.
- Security deposit and cancellation policy: [ INR Rupees 5000 and 24 hours notice for full refund ] 

TONE:
- Friendly, warm, concise. Use simple English (many guests are international).
- Never invent details about a property you don't have information on —
  say you'll confirm with the team instead of guessing.

ESCALATION — hand off to a human immediately and do NOT try to resolve
these yourself:
- Emergencies, safety issues, lockouts
- Refund or payment disputes
- Complaints about property condition
- Anything involving property damage
- Any message where the guest expresses distress, feeling unsafe, or says
  something is seriously wrong, even without an exact keyword match

When any of the above applies, start your reply with the exact tag
[ESCALATE] followed by a space, then your normal guest-facing message
telling them "I'm connecting you with our team right now, they'll reach
out within 10 minutes." Only use this tag for genuine escalations — never
for routine questions like check-in time or WiFi.

If the guest is a property owner asking about your management service,
briefly explain you offer Full Service (end-to-end management) and
Selective Service (pick specific tasks like guest messaging, cleaning
coordination, or listing optimization), then say a team member will follow
up with pricing.
"""

# ---------------------------------------------------------------------------
# 2. In-memory conversation history, keyed by guest's WhatsApp number.
#    This is fine for testing. For production, swap this dict for a real
#    database (see the note at the bottom of this file).
# ---------------------------------------------------------------------------
conversations: dict[str, list[dict]] = {}

MAX_HISTORY_MESSAGES = 20  # keep last N messages per guest to control cost


def get_claude_reply(guest_number: str, incoming_text: str) -> str:
    history = conversations.setdefault(guest_number, [])
    history.append({"role": "user", "content": incoming_text})
    history[:] = history[-MAX_HISTORY_MESSAGES:]

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=500,
        system=SYSTEM_PROMPT,
        messages=history,
    )

    reply_text = response.content[0].text
    history.append({"role": "assistant", "content": reply_text})
    return reply_text


# ---------------------------------------------------------------------------
# 3. Webhook Twilio calls whenever a WhatsApp message arrives.
#    Twilio sends form-encoded data, not JSON — hence the Form(...) params.
# ---------------------------------------------------------------------------
@app.post("/whatsapp")
async def whatsapp_webhook(
    From: str = Form(...),   # e.g. "whatsapp:+919876543210"
    Body: str = Form(...),   # the guest's message text
):
    guest_number = From
    guest_message = Body.strip()

    try:
        reply_text = get_claude_reply(guest_number, guest_message)
    except Exception as e:
        reply_text = (
            "Sorry, I'm having a technical issue right now. "
            "Our team will follow up with you shortly."
        )
        print(f"[ERROR] Claude call failed for {guest_number}: {e}")

    if reply_text.startswith("[ESCALATE]"):
        reply_text = reply_text.replace("[ESCALATE]", "", 1).strip()
        notify_manager(guest_number, guest_message, reply_text)

    twiml = MessagingResponse()
    twiml.message(reply_text)
    return PlainTextResponse(str(twiml), media_type="application/xml")


@app.get("/")
async def health_check():
    return {"status": "ok"}
