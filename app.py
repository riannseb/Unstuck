import os
import re
import time
from typing import Literal, Optional

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field


# ============================================================
# PAGE SETUP
# ============================================================

st.set_page_config(
    page_title="Unstuck",
    page_icon="⚡",
    layout="centered"
)


# ============================================================
# CONFIG
# ============================================================

# 30 seconds for hackathon demo/testing.
# Later change this to 300 for 5 minutes.
INACTIVITY_SECONDS = 30

MODEL = "gpt-5.6-luna"


# ============================================================
# SESSION STATE
# ============================================================

DEFAULT_STATE = {
    "decision": None,
    "completed_actions": [],
    "history": [],

    "barrier": None,
    "goal": None,
    "original_problem": None,

    # Clarification
    "needs_clarification": False,
    "clarification_question": None,
    "clarification_options": [],
    "status_context": None,

    # User updates during the task
    "context_updates": [],
    "show_update_box": False,

    # Stuck menu
    "show_stuck_menu": False,

    # Planning loop
    "planning_loop_detected": False,
    "planning_signals": 0,

    # Inactivity
    "action_started_at": None,
    "show_inactivity_checkin": False,
    "checkin_count": 0,

    # Completion
    "task_complete": False,
    "completion_message": None,
}


for key, value in DEFAULT_STATE.items():

    if key not in st.session_state:

        if isinstance(value, list):
            st.session_state[key] = value.copy()

        else:
            st.session_state[key] = value


# ============================================================
# OPENAI SETUP
# ============================================================

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:

    try:

        api_key = st.secrets["OPENAI_API_KEY"]

    except Exception:

        api_key = None

if not api_key:

    st.error("OpenAI API key is not configured.")

    st.stop()

client = OpenAI(api_key=api_key)


# ============================================================
# STRUCTURED OUTPUTS
# ============================================================

class UnstuckDecision(BaseModel):

    goal: str

    barrier: Literal[
        "task_overwhelm",
        "unclear_start",
        "planning_paralysis",
        "perfectionism",
        "fear_of_failure",
        "rejection_avoidance",
        "distraction",
        "low_activation",
        "other"
    ]

    needs_clarification: bool

    clarification_question: Optional[str] = None

    clarification_options: list[str] = Field(
        default_factory=list
    )

    explanation: Optional[str] = None

    next_action: Optional[str] = None

    difficulty: Optional[int] = None

    expected_minutes: Optional[int] = None


class ActionUpdate(BaseModel):

    next_action: Optional[str] = None

    explanation: str

    difficulty: Optional[int] = None

    expected_minutes: Optional[int] = None

    task_complete: bool = False

    completion_message: Optional[str] = None


# ============================================================
# CONTEXT HELPERS
# ============================================================

def get_combined_context() -> str:

    parts = []

    if st.session_state.status_context:
        parts.append(
            f"Initial status: {st.session_state.status_context}"
        )

    if st.session_state.context_updates:

        updates = "\n".join(
            f"- {update}"
            for update in st.session_state.context_updates
        )

        parts.append(
            f"Later updates from the user:\n{updates}"
        )

    if not parts:
        return "No additional status provided."

    return "\n\n".join(parts)


def add_context_update(update_text: str):

    clean_text = update_text.strip()

    if clean_text:
        st.session_state.context_updates.append(
            clean_text
        )


# ============================================================
# SAFETY FILTER
# ============================================================

HIGH_COMMITMENT_PATTERNS = [

    r"\b(click|tap|press)\s+(the\s+)?send\b",

    r"\bsend\s+(it|this|that|the message|the email)\b",

    r"\b(click|tap|press)\s+(the\s+)?submit\b",

    r"\bsubmit\s+(it|this|that|the application|the form)\b",

    r"\b(delete|erase)\s+(the|your|this|that)\b",

    r"\b(purchase|buy)\s+(the|this|that)\b",

    r"\bpost\s+(the|your|this|that)\b",

    r"\bresign\b",

    r"\bquit\s+(your|the|this)\b",

    r"\bcancel\s+(the|your|this)\b",

    r"\bpay\s+(the|this|that)\b",

    r"\btransfer\s+(the|money|funds)\b",
]


AGENCY_PHRASES = [

    "when you're ready",

    "when you are ready",

    "if you're ready",

    "if you are ready",

    "when you feel ready",

    "if you choose to",

    "you decide whether",

    "when you're comfortable",

    "when you are comfortable",
]


def is_high_commitment_action(
    action: Optional[str]
) -> bool:

    if not action:
        return False

    text = action.lower()

    # Final action is okay when agency is explicitly preserved.
    if any(
        phrase in text
        for phrase in AGENCY_PHRASES
    ):
        return False

    return any(
        re.search(pattern, text)
        for pattern in HIGH_COMMITMENT_PATTERNS
    )


def make_action_safer(
    original_problem: str,
    unsafe_action: str
) -> ActionUpdate:

    response = client.responses.parse(

        model=MODEL,

        instructions="""
You are the safety layer for an anti-procrastination app
called Unstuck.

Another model generated an action that may push the user
directly into a consequential or irreversible decision.

Examples:
- sending a message
- submitting something
- deleting something
- purchasing something
- resigning
- publicly posting
- cancelling something important
- transferring money

Your goal is NOT to stop the user from eventually taking
the final action.

Your goal is to preserve their agency.

If they are not yet ready:
give ONE reversible preparation step.

If everything genuinely appears ready:
you may bring them to the final decision point while
explicitly leaving the decision to the user.

Examples:

BAD:
"Submit it now."

GOOD:
"Everything appears ready. Press Submit when you're ready."

BAD:
"Send the message now."

GOOD:
"The message is ready. You can send it when you're comfortable."

BAD:
"Delete the file."

GOOD:
"Open the file and confirm that it is the one you intend to remove."

Rules:
- one action only
- no pressure
- no shame
- no unnecessary extra checks
- preserve user agency
""",

        input=f"""
Original problem:
{original_problem}

Potentially high-commitment action:
{unsafe_action}
""",

        text_format=ActionUpdate
    )

    return response.output_parsed


def apply_safety_filter(
    decision,
    original_problem: str
):

    if decision is None:
        return decision

    if getattr(
        decision,
        "task_complete",
        False
    ):
        return decision

    action = getattr(
        decision,
        "next_action",
        None
    )

    if (
        action
        and is_high_commitment_action(action)
    ):

        return make_action_safer(
            original_problem=original_problem,
            unsafe_action=action
        )

    return decision


# ============================================================
# PLANNING LOOP DETECTION
# ============================================================

PLANNING_PATTERNS = [

    r"\bkeep planning\b",

    r"\bkept planning\b",

    r"\bplanning instead\b",

    r"\banother plan\b",

    r"\bnew plan\b",

    r"\bre-?planning\b",

    r"\bkeep making.*plan",

    r"\bkeep making.*schedule",

    r"\bkeep changing.*schedule",

    r"\bkeep changing.*plan",

    r"\bmade.*plans?.*haven'?t\b",

    r"\bmade.*schedules?.*haven'?t\b",

    r"\bplanning but not\b",

    r"\bplan.*instead of.*doing\b",
]


def detect_planning_loop(
    text: str
) -> bool:

    if not text:
        return False

    text = text.lower()

    pattern_matches = sum(

        bool(
            re.search(
                pattern,
                text
            )
        )

        for pattern in PLANNING_PATTERNS
    )

    planning_words = [

        "plan",
        "planning",
        "schedule",
        "timetable",
        "roadmap",
        "strategy",
        "organize",
        "organise",
    ]

    avoidance_words = [

        "instead",
        "haven't started",
        "not started",
        "still haven't",
        "not doing",
        "procrastinating",
        "avoiding",
    ]

    has_planning = any(
        word in text
        for word in planning_words
    )

    has_avoidance = any(
        word in text
        for word in avoidance_words
    )

    return (
        pattern_matches >= 1
        or (
            has_planning
            and has_avoidance
        )
    )


# ============================================================
# TIMER HELPERS
# ============================================================

def start_action_timer():

    st.session_state.action_started_at = (
        time.time()
    )

    st.session_state.show_inactivity_checkin = False


# ============================================================
# STORE ACTION / COMPLETION
# ============================================================

def set_action_decision(
    decision
):

    if decision is None:
        return

    # --------------------------------------------------------
    # TASK COMPLETE
    # --------------------------------------------------------

    if getattr(
        decision,
        "task_complete",
        False
    ):

        st.session_state.task_complete = True

        st.session_state.completion_message = (
            decision.completion_message
            or
            "You did it. Take the win. 🎉"
        )

        st.session_state.decision = None

        st.session_state.action_started_at = None

        st.session_state.show_inactivity_checkin = False

        st.session_state.show_stuck_menu = False

        st.session_state.show_update_box = False

        return

    # --------------------------------------------------------
    # NORMAL ACTION
    # --------------------------------------------------------

    safe_decision = apply_safety_filter(

        decision,

        st.session_state.original_problem
        or ""
    )

    st.session_state.decision = safe_decision

    st.session_state.task_complete = False

    st.session_state.completion_message = None

    start_action_timer()


# ============================================================
# INITIAL DECISION
# ============================================================

def get_unstuck_decision(
    problem: str
):

    response = client.responses.parse(

        model=MODEL,

        instructions="""
You are the decision engine for an anti-procrastination app
called Unstuck.

Your goal is to help the user move from intention to action.

Do NOT blindly produce a micro-task.

First determine whether you understand WHERE THE USER CURRENTLY IS
relative to their goal.

--------------------------------------------------
WHEN TO ASK A CLARIFICATION QUESTION
--------------------------------------------------

If the goal is broad and the correct next action depends
on how much the user has already completed, ask ONE short
clarification question first.

Examples:

"I need to submit my hackathon."

"I need to finish my assignment."

"I need to prepare for an interview."

"I need to apply for jobs."

"I need to clean my house."

"I need to finish my thesis."

Example:

User:
"I need to submit my hackathon."

Better:
"Where are you with the submission right now?"

Possible answers:
- I haven't started preparing it
- The project is still being built
- The project works but the README/demo is incomplete
- Everything is ready and I just need to upload/submit
- I'm not sure what is left

Do not ask unnecessary questions.

If enough context already exists, start immediately.

--------------------------------------------------
BLOCKERS
--------------------------------------------------

Choose one:

- task_overwhelm
- unclear_start
- planning_paralysis
- perfectionism
- fear_of_failure
- rejection_avoidance
- distraction
- low_activation
- other

--------------------------------------------------
PLANNING PARALYSIS
--------------------------------------------------

If the user repeatedly:
- plans
- schedules
- reorganises
- researches how to begin

without actually acting,

classify it as planning_paralysis.

Do NOT give them another large plan.

Move toward execution.

--------------------------------------------------
FIRST ACTION
--------------------------------------------------

If clarification is unnecessary:

Give exactly ONE useful next action.

The first action should usually:
- be concrete
- be observable
- take roughly 1–5 minutes
- reduce activation energy
- not be a checklist
- not create another plan
- not shame the user
- preserve agency

Avoid forcing consequential actions.
""",

        input=problem,

        text_format=UnstuckDecision
    )

    return response.output_parsed


# ============================================================
# AFTER CLARIFICATION
# ============================================================

def get_action_after_clarification(
    original_problem: str,
    clarification_question: str,
    clarification_answer: str
):

    response = client.responses.parse(

        model=MODEL,

        instructions="""
You are the action engine for Unstuck.

The user gave a broad goal.

You asked where they currently are.

They have now told you their status.

Give exactly ONE useful next action based on their ACTUAL
current stage.

Do not send them back through things they have already done.

Rules:
- one meaningful action
- concrete
- observable
- preferably 1–5 minutes initially
- no unnecessary checklist
- no unnecessary verification
- no long plan
- do not repeat work
- preserve agency
- do not invent steps merely to prolong the process

The explanation should be brief.
""",

        input=f"""
Original goal:
{original_problem}

Clarification question:
{clarification_question}

Current status:
{clarification_answer}
""",

        text_format=ActionUpdate
    )

    return response.output_parsed


# ============================================================
# NEXT ACTION
# ============================================================

def get_next_action(
    original_problem: str,
    current_action: str,
    feedback: str,
    completed_actions: list[str],
    status_context: Optional[str] = None
):

    completed_text = (

        "\n".join(
            f"- {x}"
            for x in completed_actions
        )

        if completed_actions

        else "None yet"
    )

    completed_count = len(
        completed_actions
    )

    response = client.responses.parse(

        model=MODEL,

        instructions="""
You are the adaptive action engine for Unstuck.

Your goal is NOT to keep the user doing tiny steps forever.

Your goal is:

START
→ BUILD MOMENTUM
→ MAKE MEANINGFUL PROGRESS
→ FINISH

--------------------------------------------------
WHEN FEEDBACK = "done"
--------------------------------------------------

The user successfully completed the current action.

Give a brief, natural acknowledgement.

Examples:

"Good — you're moving now."

"Nice. You've got momentum; let's use it."

"That's one less thing between you and finishing."

Do NOT excessively praise every click.

Then determine whether another action is actually needed.

--------------------------------------------------
TASK COMPLETION
--------------------------------------------------

After every "done", determine whether the user's ORIGINAL GOAL
has actually been completed.

If YES:

task_complete = true

next_action = null

Give a short, sincere completion message.

Acknowledge:
1. they completed the task
2. they moved through resistance/procrastination

Examples:

"You did it 🎉 You went from being stuck to actually finishing it.
Take the win."

"Done. You kept moving even when avoiding it would have been easier."

"That's finished 🎉 You didn't wait until you felt perfectly ready —
you got moving and followed through."

Do NOT generate another task.

Do NOT immediately turn success into another productivity goal.

If NOT complete:

task_complete = false

Generate the next action.

--------------------------------------------------
ADAPT ACTION SIZE
--------------------------------------------------

completed_count == 0:

- small activation step
- approximately 1–5 minutes

completed_count == 1 or 2:

- slightly broader action
- combine obvious adjacent steps
- 5–10 minutes is okay

completed_count >= 3:

- assume momentum exists
- stop baby-stepping unnecessarily
- combine straightforward steps
- 10–15 minutes is acceptable
- move strongly toward completion

The progression should be:

tiny step
→ normal work
→ meaningful progress
→ completion

--------------------------------------------------
DO NOT INVENT USELESS STEPS
--------------------------------------------------

Avoid:
- checking whether a page loaded
- looking for a button that is obviously there
- taking screenshots without a reason
- rereading information already understood
- checking warnings that have not appeared
- reopening something already open
- verifying obvious interface details

Every step must meaningfully reduce the remaining distance
to the original goal.

--------------------------------------------------
NEAR COMPLETION
--------------------------------------------------

If preparation is complete, move directly to the final
decision point.

Do not invent extra preparation.

Acceptable:

"Everything appears ready. Press Submit when you're ready."

"The message is ready. Send it when you're comfortable."

"The application is ready. Submit it when you're ready."

Preserve user agency.

--------------------------------------------------
IF FEEDBACK = "too_hard"
--------------------------------------------------

Make the action meaningfully easier.

- reduce effort
- reduce commitment
- make it more concrete
- do not simply rephrase the same task
- returning to a 1–5 minute step is okay

--------------------------------------------------
GENERAL
--------------------------------------------------

Generate at most ONE action.

No long lists.

No unnecessary planning.

No motivational speeches.

Progress should feel efficient.
""",

        input=f"""
Original goal:
{original_problem}

Known current status and later updates:
{status_context or "No additional status"}

Current action:
{current_action}

Feedback:
{feedback}

Number of actions completed:
{completed_count}

Completed actions:
{completed_text}
""",

        text_format=ActionUpdate
    )

    return response.output_parsed


# ============================================================
# USER CONTEXT UPDATE
# ============================================================

def handle_context_update(
    original_problem: str,
    current_action: str,
    update_text: str,
    completed_actions: list[str],
    existing_context: str
):

    completed_text = (

        "\n".join(
            f"- {x}"
            for x in completed_actions
        )

        if completed_actions

        else "None yet"
    )

    response = client.responses.parse(

        model=MODEL,

        instructions="""
You are updating your understanding of a user who is currently
working toward a goal with Unstuck.

The user has provided NEW INFORMATION.

This may mean:
- they completed more work than you realised
- they skipped ahead
- they encountered an intermediate step
- something changed
- your previous assumption was wrong
- the task has changed slightly
- they need to explain what is happening in the real world

Treat the user's newest update as authoritative.

Do NOT force them through outdated steps.

Do NOT insist that they follow your previous sequence.

Reassess where they are NOW.

Then give exactly ONE useful next action from their real
current position.

If their update indicates that the original goal is already complete:

task_complete = true

Otherwise:

task_complete = false

and generate one next action.

The explanation should briefly acknowledge what changed.

Rules:
- no unnecessary replanning
- no obsolete steps
- no repetition
- no long checklist
- skip ahead when appropriate
- preserve agency
- move meaningfully toward the original goal
""",

        input=f"""
Original goal:
{original_problem}

Previously suggested action:
{current_action}

Existing context:
{existing_context}

NEW INFORMATION FROM USER:
{update_text}

Previously completed actions:
{completed_text}
""",

        text_format=ActionUpdate
    )

    return response.output_parsed


# ============================================================
# I'M STUCK FLOW
# ============================================================

def handle_stuck_reason(
    original_problem: str,
    current_action: str,
    stuck_reason: str,
    completed_actions: list[str],
    context: str
):

    completed_text = (

        "\n".join(
            f"- {x}"
            for x in completed_actions
        )

        if completed_actions

        else "None yet"
    )

    response = client.responses.parse(

        model=MODEL,

        instructions="""
You are the supportive action companion inside Unstuck.

The user has said they are stuck.

Your response has THREE goals:

1. CALM
Respond to what they are actually experiencing.

2. ENCOURAGE
Sound like a grounded person who genuinely wants them to finish.

3. MOVE
Give exactly ONE concrete next action.

--------------------------------------------------
IMPORTANT TONE
--------------------------------------------------

Do not sound like a motivational poster.

Do not give random inspirational quotes.

Do not say things like:

"You've got this!!!"

"You're amazing!"

"Believe in yourself!"

Instead, say something SPECIFIC to the user's difficulty.

The support should feel like someone sitting beside them,
helping them through the moment.

Use approximately 1–3 short sentences of support.

Then give ONE action.

--------------------------------------------------
OVERWHELM
--------------------------------------------------

Possible tone:

"You don't need to hold the whole task in your head right now.
Nothing beyond the next move needs your attention yet."

"We can make the task smaller without giving up on it.
You only need enough energy for the piece directly in front of you."

Then reduce action size.

--------------------------------------------------
PERFECTIONISM / FEAR OF DOING IT BADLY
--------------------------------------------------

Possible tone:

"It doesn't need to be impressive yet. It only needs to exist
before you can improve it."

"You're allowed to make a rough version. Finishing imperfectly
gets you much further than polishing something that never gets done."

Then give a rough/reversible action.

--------------------------------------------------
FEAR OF REJECTION / SOMEONE'S REACTION
--------------------------------------------------

Possible tone:

"The uncertainty about their reaction is making this feel bigger
than the action itself. You don't have to know what they will think
before you move."

"You only control your side of this interaction. We can take one
step without deciding the whole outcome."

Never claim another person will react positively.

--------------------------------------------------
DOESN'T KNOW WHAT TO DO
--------------------------------------------------

Possible tone:

"You don't need the entire route figured out. We only need to make
the next few minutes obvious."

Then give one clear next move.

--------------------------------------------------
DISTRACTED
--------------------------------------------------

Possible tone:

"Getting distracted doesn't erase the progress you've already made.
You don't need to restart — just come back in where you left."

Then make re-entry easy.

--------------------------------------------------
CAN'T MAKE THEMSELVES START
--------------------------------------------------

Possible tone:

"You don't need to suddenly feel motivated. We only need enough
movement to break the standstill."

"Don't ask yourself to finish it yet. Let's only get your body
and attention into the starting position."

Then reduce activation energy.

--------------------------------------------------
CUSTOM / OTHER REASON
--------------------------------------------------

If the user writes their own explanation:

Read it carefully.

Respond to THEIR actual issue rather than forcing it into one of
the predefined categories.

They may tell you:
- they already progressed much further
- there is an unexpected intermediate step
- circumstances changed
- the task is emotionally difficult in another way
- the previous suggested action doesn't fit reality

Adapt accordingly.

If their message is primarily a factual progress update,
skip obsolete actions and guide them from their actual current stage.

--------------------------------------------------
SAFETY
--------------------------------------------------

If high-stakes social or consequential actions are involved,
preserve user agency.

Do not pressure them into:
- sending
- submitting
- posting
- deleting
- quitting
- purchasing

A final action is okay when framed as their choice:

"Send it when you're ready."

--------------------------------------------------
OUTPUT
--------------------------------------------------

The explanation should contain the calming/supportive message.

The next_action must contain exactly ONE concrete action.

The action should:
- be observable
- be appropriate for their actual current position
- usually take 1–5 minutes when they are stuck
- not be a long list
- not diagnose them
- not make claims about someone else's thoughts

task_complete should normally be false unless the user's
new information clearly shows the goal is already complete.
""",

        input=f"""
Original goal:
{original_problem}

Current action:
{current_action}

Known context:
{context}

What is making the user stuck:
{stuck_reason}

Completed actions:
{completed_text}
""",

        text_format=ActionUpdate
    )

    return response.output_parsed


# ============================================================
# INACTIVITY RESPONSE
# ============================================================

def handle_inactivity(
    current_action: str,
    reason: str,
    context: str
):

    if reason == "I'm doing it":

        start_action_timer()

        return None

    feedback_map = {

        "I got distracted":
        """
The user's attention wandered.
Help them return without restarting everything.
""",

        "It's too hard":
        """
The current action feels too difficult.
Meaningfully reduce its size.
""",

        "I'm avoiding it":
        """
The user appears to be avoiding the current action.
Reduce emotional commitment and make approaching the task easier.
"""
    }

    response = client.responses.parse(

        model=MODEL,

        instructions="""
You are handling an inactivity check-in for Unstuck.

The user was given an action but did not respond.

Your response should contain:

1. ONE brief supportive/calming response.
2. ONE concrete action.

Do not shame them.

Do not make them restart everything.

Tone examples:

DISTRACTED:
"No problem — wandering away doesn't erase your progress.
Let's just step back in."

TOO HARD:
"That's useful information, not failure. The step was still
too large, so we'll make it smaller."

AVOIDING:
"If you're avoiding it, there's probably some friction here.
We don't need to overpower it — let's lower the commitment."

Then provide ONE action.

Rules:
- one action
- no lecture
- no huge plan
- easy re-entry
- preserve agency
- usually 1–5 minutes
""",

        input=f"""
Current action:
{current_action}

Known user context:
{context}

What happened:
{feedback_map[reason]}
""",

        text_format=ActionUpdate
    )

    return response.output_parsed


# ============================================================
# RESET
# ============================================================

def reset_session():

    for key, value in DEFAULT_STATE.items():

        if isinstance(
            value,
            list
        ):

            st.session_state[key] = (
                value.copy()
            )

        else:

            st.session_state[key] = (
                value
            )


# ============================================================
# UI
# ============================================================

st.title(
    "Un⚡tuck"
)

st.subheader(
    "From procrastination to one doable action"
)

st.write(
    "Tell me what you need to do and what's stopping you from it."
)


# ============================================================
# COMPLETION SCREEN
# ============================================================

if st.session_state.task_complete:

    st.success(
        "🎉 Task complete"
    )

    st.markdown(
        f"### {st.session_state.completion_message}"
    )

    st.write(
        "Take the win. You don't need to turn it into another task."
    )


    if st.session_state.completed_actions:

        with st.expander(
            "See what you accomplished"
        ):

            for action in (
                st.session_state.completed_actions
            ):

                st.write(
                    f"✓ {action}"
                )


    if st.button(
        "Start something else"
    ):

        reset_session()

        st.rerun()


# ============================================================
# NORMAL APP
# ============================================================

else:

    if (
        st.session_state.decision is not None
        or st.session_state.needs_clarification
    ):

        if st.button(
            "↻ Start over"
        ):

            reset_session()

            st.rerun()


    problem = st.text_area(

        "What are you stuck on?",

        placeholder=(
            "I need to submit my hackathon, "
            "but I'm not sure what I should do next..."
        )
    )


    # ========================================================
    # START
    # ========================================================

    if st.button(
        "Help me start"
    ):

        if not problem:

            st.warning(
                "Tell me what you're stuck on first."
            )

        else:

            with st.spinner(
                "Figuring out where you are and what would help next..."
            ):

                try:

                    decision = (
                        get_unstuck_decision(
                            problem
                        )
                    )

                    # New goal = fresh state
                    st.session_state.completed_actions = []

                    st.session_state.context_updates = []

                    st.session_state.show_stuck_menu = False

                    st.session_state.show_update_box = False

                    st.session_state.original_problem = (
                        problem
                    )

                    st.session_state.task_complete = False

                    st.session_state.completion_message = None

                    st.session_state.status_context = None

                    st.session_state.goal = (
                        decision.goal
                    )

                    st.session_state.barrier = (
                        decision.barrier
                    )


                    planning_loop = (

                        detect_planning_loop(
                            problem
                        )

                        or

                        decision.barrier
                        ==
                        "planning_paralysis"
                    )

                    st.session_state.planning_loop_detected = (
                        planning_loop
                    )


                    if planning_loop:

                        st.session_state.planning_signals += 1


                    # ----------------------------------------
                    # Need clarification
                    # ----------------------------------------

                    if decision.needs_clarification:

                        st.session_state.needs_clarification = True

                        st.session_state.clarification_question = (
                            decision.clarification_question
                        )

                        st.session_state.clarification_options = (
                            decision.clarification_options
                        )

                        st.session_state.decision = None


                    # ----------------------------------------
                    # Can act immediately
                    # ----------------------------------------

                    else:

                        st.session_state.needs_clarification = False

                        set_action_decision(
                            decision
                        )


                    st.rerun()


                except Exception as e:

                    st.error(
                        f"Something went wrong: {e}"
                    )


    # ========================================================
    # CLARIFICATION
    # ========================================================

    if st.session_state.needs_clarification:

        st.success(
            "Let me understand where you are first."
        )


        if st.session_state.planning_loop_detected:

            st.warning(
                "🔁 Planning pattern detected — "
                "I'll avoid giving you another big plan."
            )


        st.markdown(
            f"### {st.session_state.clarification_question}"
        )


        options = (
            st.session_state.clarification_options
        )


        if options:

            clarification_answer = st.radio(

                "Choose the closest option:",

                options,

                index=None
            )

        else:

            clarification_answer = st.text_area(
                "Tell me where you're up to:"
            )


        if clarification_answer:

            if st.button(
                "Continue",
                key="clarification_continue"
            ):

                with st.spinner(
                    "Finding the next useful action..."
                ):

                    try:

                        st.session_state.status_context = (
                            clarification_answer
                        )

                        action = (
                            get_action_after_clarification(

                                original_problem=(
                                    st.session_state.original_problem
                                ),

                                clarification_question=(
                                    st.session_state.clarification_question
                                ),

                                clarification_answer=(
                                    clarification_answer
                                )
                            )
                        )

                        st.session_state.needs_clarification = False

                        set_action_decision(
                            action
                        )

                        st.rerun()


                    except Exception as e:

                        st.error(
                            f"Something went wrong: {e}"
                        )


    # ========================================================
    # MAIN ACTION UI
    # ========================================================

    if (

        not st.session_state.needs_clarification

        and

        st.session_state.decision is not None

    ):

        decision = (
            st.session_state.decision
        )


        st.success(
            "Let's make this easier."
        )


        if st.session_state.barrier:

            st.caption(
                "Likely blocker: "
                f"{st.session_state.barrier.replace('_', ' ').title()}"
            )


        if st.session_state.planning_loop_detected:

            st.warning(
                "🔁 Planning loop detected — "
                "we're switching from planning to action."
            )

        st.markdown(
                "### 🧙🏻‍♂️💭 Yodha says  "
            )

        if decision.explanation:

            st.write(
                decision.explanation
            )


        st.markdown(
            "### 🎯Your next step"
        )


        st.info(
            decision.next_action
        )


        if decision.expected_minutes is not None:

            st.caption(
                f"About {decision.expected_minutes} minute(s)"
            )


        # ====================================================
        # MAIN BUTTONS
        # ====================================================

        col1, col2, col3, col4 = (
            st.columns(4)
        )


        with col1:

            done = st.button(
                "✅ Done",
                use_container_width=True
            )


        with col2:

            too_hard = st.button(
                "🤯 Too hard",
                use_container_width=True
            )


        with col3:

            stuck = st.button(
                "😵‍💫 I'm stuck",
                use_container_width=True
            )


        with col4:

            update = st.button(
                "☝️🤓 Update me",
                use_container_width=True
            )


        # ====================================================
        # DONE
        # ====================================================

        if done:

            st.session_state.completed_actions.append(
                decision.next_action
            )


            with st.spinner(
                "Checking what's left..."
            ):

                new_decision = get_next_action(

                    original_problem=(
                        st.session_state.original_problem
                    ),

                    current_action=(
                        decision.next_action
                    ),

                    feedback="done",

                    completed_actions=(
                        st.session_state.completed_actions
                    ),

                    status_context=(
                        get_combined_context()
                    )
                )


            set_action_decision(
                new_decision
            )

            st.rerun()


        # ====================================================
        # TOO HARD
        # ====================================================

        if too_hard:

            with st.spinner(
                "Making the step smaller..."
            ):

                new_decision = get_next_action(

                    original_problem=(
                        st.session_state.original_problem
                    ),

                    current_action=(
                        decision.next_action
                    ),

                    feedback="too_hard",

                    completed_actions=(
                        st.session_state.completed_actions
                    ),

                    status_context=(
                        get_combined_context()
                    )
                )


            set_action_decision(
                new_decision
            )

            st.rerun()


        # ====================================================
        # I'M STUCK
        # ====================================================

        if stuck:

            st.session_state.show_stuck_menu = True

            st.session_state.show_update_box = False

            st.session_state.show_inactivity_checkin = False

            st.rerun()


        # ====================================================
        # UPDATE ME
        # ====================================================

        if update:

            st.session_state.show_update_box = True

            st.session_state.show_stuck_menu = False

            st.session_state.show_inactivity_checkin = False

            st.rerun()


        # ====================================================
        # USER UPDATE BOX
        # ====================================================

        if st.session_state.show_update_box:

            st.markdown(
                "### ✏️ Tell me what changed"
            )

            st.write(
                "You can correct me, tell me you've moved ahead, "
                "mention an unexpected step, or add anything I should "
                "know before choosing what comes next."
            )


            update_text = st.text_area(

                "What's different now?",

                placeholder=(
                    "For example: I already finished the README "
                    "and demo. I'm now on the Devpost submission page."
                ),

                key="context_update_text"
            )


            update_col1, update_col2 = (
                st.columns(2)
            )


            with update_col1:

                use_update = st.button(
                    "Use this update",
                    key="use_context_update",
                    use_container_width=True
                )


            with update_col2:

                cancel_update = st.button(
                    "Cancel",
                    key="cancel_context_update",
                    use_container_width=True
                )


            if cancel_update:

                st.session_state.show_update_box = False

                st.rerun()


            if use_update:

                if not update_text.strip():

                    st.warning(
                        "Tell me what changed first."
                    )

                else:

                    add_context_update(
                        update_text
                    )


                    with st.spinner(
                        "Updating where we are..."
                    ):

                        new_decision = handle_context_update(

                            original_problem=(
                                st.session_state.original_problem
                            ),

                            current_action=(
                                decision.next_action
                            ),

                            update_text=(
                                update_text
                            ),

                            completed_actions=(
                                st.session_state.completed_actions
                            ),

                            existing_context=(
                                get_combined_context()
                            )
                        )


                    st.session_state.show_update_box = False

                    set_action_decision(
                        new_decision
                    )

                    st.rerun()


        # ====================================================
        # STUCK MENU
        # ====================================================

        if st.session_state.show_stuck_menu:

            st.markdown(
                "### 💭 What's making this hard right now?"
            )


            blocker_choice = st.radio(

                "Choose the closest option:",

                [
                    "It feels overwhelming",

                    "I'm afraid I'll do it badly",

                    "I'm worried about someone's reaction",

                    "I don't know what to do",

                    "I got distracted",

                    "I just can't make myself start",

                    "Other — I'll explain"
                ],

                index=None
            )


            custom_stuck_text = ""


            # ------------------------------------------------
            # OTHER / CUSTOM ISSUE
            # ------------------------------------------------

            if blocker_choice == "Other — I'll explain":

                custom_stuck_text = st.text_area(

                    "Tell me what's happening:",

                    placeholder=(
                        "For example: I already did more than the app "
                        "realises, there's another step in between, "
                        "something changed, or I'm stuck for a different reason..."
                    ),

                    key="custom_stuck_text"
                )


            # Optional extra detail for ANY option
            elif blocker_choice:

                extra_detail = st.text_area(

                    "Anything else you want me to know? (optional)",

                    placeholder=(
                        "You can add context about what you're feeling, "
                        "what changed, or how far you've already got."
                    ),

                    key="stuck_extra_detail"
                )

            else:

                extra_detail = ""


            # ------------------------------------------------
            # HELP BUTTON
            # ------------------------------------------------

            if blocker_choice:

                if st.button(
                    "Help me through this",
                    key="stuck_help"
                ):

                    if (
                        blocker_choice
                        ==
                        "Other — I'll explain"
                        and
                        not custom_stuck_text.strip()
                    ):

                        st.warning(
                            "Tell me what's getting in the way first."
                        )

                    else:

                        if (
                            blocker_choice
                            ==
                            "Other — I'll explain"
                        ):

                            stuck_reason = (
                                custom_stuck_text.strip()
                            )

                        else:

                            stuck_reason = (
                                blocker_choice
                            )

                            if extra_detail.strip():

                                stuck_reason += (
                                    "\n\nAdditional information from user:\n"
                                    +
                                    extra_detail.strip()
                                )


                        # Save useful free-text information
                        if (
                            blocker_choice
                            ==
                            "Other — I'll explain"
                        ):

                            add_context_update(
                                custom_stuck_text
                            )

                        elif extra_detail.strip():

                            add_context_update(
                                extra_detail
                            )


                        with st.spinner(
                            "I'm with you — let's work through this..."
                        ):

                            new_decision = handle_stuck_reason(

                                original_problem=(
                                    st.session_state.original_problem
                                ),

                                current_action=(
                                    decision.next_action
                                ),

                                stuck_reason=(
                                    stuck_reason
                                ),

                                completed_actions=(
                                    st.session_state.completed_actions
                                ),

                                context=(
                                    get_combined_context()
                                )
                            )


                        st.session_state.show_stuck_menu = False

                        set_action_decision(
                            new_decision
                        )

                        st.rerun()


        # ====================================================
        # MOMENTUM
        # ====================================================

        if st.session_state.completed_actions:

            st.markdown(
                "### 🔥 Momentum"
            )


            for action in (
                st.session_state.completed_actions[-3:]
            ):

                st.write(
                    f"✓ {action}"
                )


# ============================================================
# INACTIVITY CHECK-IN
# ============================================================

@st.fragment(
    run_every="2s"
)
def inactivity_monitor():

    if st.session_state.task_complete:
        return

    if st.session_state.decision is None:
        return

    if st.session_state.needs_clarification:
        return

    if st.session_state.show_stuck_menu:
        return

    if st.session_state.show_update_box:
        return


    started_at = (
        st.session_state.action_started_at
    )


    if started_at is None:
        return


    elapsed = (
        time.time()
        -
        started_at
    )


    if (
        elapsed
        >=
        INACTIVITY_SECONDS
    ):

        st.session_state.show_inactivity_checkin = True


    if st.session_state.show_inactivity_checkin:

        st.divider()

        st.markdown(
            "### 👋 Still there?"
        )

        st.write(
            "No pressure — I'm still here. What happened?"
        )


        c1, c2 = st.columns(2)


        with c1:

            doing_it = st.button(
                "I'm doing it 🤩",
                key="inactive_doing",
                use_container_width=True
            )


            distracted = st.button(
                "I got distracted 😅",
                key="inactive_distracted",
                use_container_width=True
            )


        with c2:

            too_difficult = st.button(
                "It's too hard 🤔💭",
                key="inactive_hard",
                use_container_width=True
            )


            avoiding = st.button(
                "I'm avoiding it 🫣",
                key="inactive_avoiding",
                use_container_width=True
            )


        # ----------------------------------------------------
        # STILL WORKING
        # ----------------------------------------------------

        if doing_it:

            st.session_state.checkin_count += 1

            start_action_timer()

            st.rerun()


        # ----------------------------------------------------
        # DISTRACTED
        # ----------------------------------------------------

        if distracted:

            with st.spinner(
                "Let's get you back into it..."
            ):

                new_decision = handle_inactivity(

                    current_action=(
                        st.session_state.decision.next_action
                    ),

                    reason="I got distracted",

                    context=(
                        get_combined_context()
                    )
                )


            st.session_state.checkin_count += 1

            set_action_decision(
                new_decision
            )

            st.rerun()


        # ----------------------------------------------------
        # TOO HARD
        # ----------------------------------------------------

        if too_difficult:

            with st.spinner(
                "Let's make this lighter..."
            ):

                new_decision = handle_inactivity(

                    current_action=(
                        st.session_state.decision.next_action
                    ),

                    reason="It's too hard",

                    context=(
                        get_combined_context()
                    )
                )


            st.session_state.checkin_count += 1

            set_action_decision(
                new_decision
            )

            st.rerun()


        # ----------------------------------------------------
        # AVOIDING
        # ----------------------------------------------------

        if avoiding:

            with st.spinner(
                "Let's lower the pressure..."
            ):

                new_decision = handle_inactivity(

                    current_action=(
                        st.session_state.decision.next_action
                    ),

                    reason="I'm avoiding it",

                    context=(
                        get_combined_context()
                    )
                )


            st.session_state.checkin_count += 1

            set_action_decision(
                new_decision
            )

            st.rerun()


inactivity_monitor()