# Coding guide

Appendix A of the thesis *Automated Turn-Level Classification of Student Interaction Roles in Student-LLM Programming Conversations* (Donghang Wen, Bachelor of Advanced Computing (Honours), The University of Sydney, 2026). Section and table numbers refer to the thesis.

This appendix gives the coding guide as revised after the discussion between the coders (Section 3.4.3). It states 17 rules, each a test to apply to the visible text. The rules were applied to the double-coded set and the first additional set, the researcher coded the second additional set with them, and the prompted model received them (Section 3.5.4). The second coder's independent coding of the double-coded set used the earlier guide of 16 October 2026. Table 3.2 lists the dimensions and values. The examples come from the double-coded set, except one from the calibration material, and give the code in the ground truth. They are described here; the guide given to the coders and to the prompted model named them by identifier.

## A.1 Reading a target

A target is one student prompt and the assistant response that follows it. Read the conversation in order and use earlier turns to understand what the student is doing now. A displayed gap in the turn numbers means missing context; do not read across it.

**Rule 1. Code what the turn does in its context, not how it is worded.** Question words, imperative verbs, the presence of code, the length of the response and phrases such as "Let me know" do not decide a code by themselves. Code what the student asked for, not what the assistant gave. A later exchange does not change the code of an earlier one. Turns with similar wording can take different codes when their evidence differs.

**Rule 2. Every code needs something you can point to.** Student codes need something the student said or did: a request, a proposal, an attempt or a condition, read with the earlier turns. The two AI dimensions are coded from the current response; earlier turns only show what it responds to. The assistant's response is not evidence of what the student meant, or of what an attachment that was not preserved contained.

## A.2 Student interaction type (select one)

Ask these questions in order and stop at the first that applies.

1. Is there a request, either in this turn or continued from a help task visible in earlier turns (Rule 3)? If not, select Providing material / responding.

2. Does the student put forward their own plan, choice, condition or attempt for the assistant to check or develop, or use it to redirect the task (Rules 4 and 5)? If so, select Guided generation.

3. Does the student ask the assistant to produce, change, investigate, debug, check or continue work (Rule 6)? If so, select Task automation.

4. Otherwise the student asks for information, a method or a meaning: select Information retrieval.

**Rule 3. A request can continue from earlier turns.** When the visible earlier turns show a concrete help task, a report that the fault persists, that a supplied step failed or that a new obstacle arose continues that request: Task automation. This holds for a pasted log without a question, and when the assistant asked for the output. Asking the assistant to continue, to check that the work is complete, or to correct its own misreading is also Task automation. Output that shows success, an answer to the assistant's question, thanks, and attachments with only a statement of scope are Providing material / responding. With no visible earlier context (an opening turn, or the turn after a gap), a bare paste with no words of the student's own is Providing material / responding; flag the concrete difficulty. A fault stated in the student's own words is a request even in an opening turn.

Examples:

- A student pastes the output of a key generation that worked: Providing material / responding. In the next interaction the same student reports that the next supplied step failed with "No such file or directory": Task automation.
- An opening turn with code and a sanitizer report and no words of the student's own: Providing material / responding, flagged. An opening turn saying that the client still does not exit: Task automation.

**Rule 4. Guided generation needs the student's own contribution.** The contribution is a plan, choice, condition, diagnosis or attempt that the student puts forward, not a restatement of what the assistant has said. It can appear in earlier turns, and a proposed design need not be implemented yet. A follow-up that develops or checks the student's earlier proposal stays Guided generation. An attempt needs evidence, such as a described test or result; presenting a file does not by itself show an attempt. If the student has done part of the work and asks the assistant to evaluate or continue it, the turn is Guided generation even when the reply is a complete answer. Simply accepting an assistant suggestion is not enough; turning it down for an alternative of the student's own choosing redirects the task. A comprehension check, a knowledge question, or asking what a tool does is Information retrieval even when the student mentions their own project or intention.

Examples:

- A student sets out their own plan for collecting and applying edits and asks about it: Guided generation.
- A student checks their understanding of what the STAILQ head corresponds to: Information retrieval.

**Rule 5. Conditions from the assignment are not the student's design.** Conditions copied from the assignment, including its hints, do not make a request Guided generation. Rewriting a requirement in the student's own words can show judgement, and the student's own condition still counts when it is combined with assignment or assistant material. If the task is clear but the origin of a condition cannot be established, select Task automation and flag the default.

Examples:

- A student asks how to implement the one-thread-per-client model, which the assignment requires: not Guided generation.

**Rule 6. A method is information; a product is a task.** Asking whether or how something can be done, what a good way is, what something means or how a mechanism works is Information retrieval. Asking the assistant to write, change or produce something, including "show me a code example", is Task automation. A failure log or failing tests with a question, or asking why a specific program fails, asks for troubleshooting: Task automation. An explicit request to remember code that is about to be supplied is Task automation. Rule 4 is checked first.

Examples:

- A student asks how to handle cursor positions after text is deleted: Information retrieval.
- A student asks for a C code example: Task automation.

## A.3 Topic (select every supported topic)

**Rule 7. Choose the topic by what the student asks about.** Code: the student's own program, what a piece of code does, how to change it, or how to implement the assignment's protocol or algorithm in it, even with no code pasted. A known assignment document, or code the assistant wrote earlier, can be that object, and a repair approach for that code is code. Concept: how a language feature, library function, system mechanism or algorithm works as such, apart from the student's program. Select both only when the student asks about both. The wording "how to implement" or "explain" does not decide. Do not take a topic from the assistant's reply or from the interaction type, and do not select two topics to express doubt.

Examples:

- A student asks how sigemptyset, sigaddset and sigprocmask are used to block signals: concept. Another asks how to write the qsort comparison with their own variable names: code.
- A student asks whether their own client can receive the signal before sigwait is called, which turns on how blocked signals behave: code and concept.

**Rule 8. Tooling or environment needs the tool itself as an object.** Select it when the tool, build configuration, repository or execution environment is what the student asks about; the student need not name the tool. Writing a Makefile is tooling; organising the tests for a function is code. A tool named in output, such as a sanitizer report, a tool introduced by the assistant, or a test file name does not add tooling. A link failure in a build log is a failure of the build step, so tooling applies even when the student only asks what the error means. Tooling and code can both apply.

Examples:

- A sanitizer report about the student's code: code only. A link failure in the build log: code and tooling.

**Rule 9. Conversation management is about the conversation itself.** Select it for thanking and closing, asking for a summary or review of the conversation, asking the assistant to remember or ignore earlier turns, or asking it to change how it answers. Answering the assistant, adding information, supplying material or stating the scope of the work is not conversation management; the topic is what that information is about.

Examples:

- A student asks for a summary of what the assistant was used for: conversation management. Another tells the assistant which folder an object file is in: tooling, not conversation management.

## A.4 AI delegation (select every value that applies; ND alone)

Table A.1 gives the meaning of each value.

**Table A.1** Values of AI delegation

| Code | Meaning |
|---|---|
| ND | Neither an offer nor a hand-back. |
| offer | Offers identifiable further work by the assistant. |
| hand-back: directive | Leaves the student work that the response does not supply. |
| hand-back: asks the student to supply | Asks for information, code, output, status or material. |
| hand-back: Socratic | Asks the student to reason through a question or explain their reasoning. |

**Rule 10. Directive means work left to the student beyond the answer.** Directive applies when the response leaves the student an implementation it does not supply, an actual check or test of their real state, or a decision. A check of the real state is directive even inside the answer and even when the command is supplied, and a conditional sentence can still assign work. When the student asks how to do something and the response describes how, answers the student's hypothetical question, or gives the plan the student asked for, that is the answer, not directive. Corrected code, a command to run, instructions for using the answer, a list of alternatives, what an option would involve, recommendations of good practice and general handling rules are not directive. Wording such as "make sure" does not decide.

Examples:

- A response asks the student to double-check how their client opens the FIFOs: directive. In a calibration example, a response supplies the corrected header and leaves nothing else: ND.

**Rule 11. An offer invites more help; asks to supply requests something from the student.** An invitation to ask for or accept identifiable further help is an offer wherever it appears, including a conditional one: "Let me know if you get stuck" or "Let me know if you get an error". A request for information, code, output or status that the student is to provide is asks the student to supply: "Paste the output" or "Let me know when it finishes". One sentence can do both: "Let me know when you've finished implementing X, then I can help with Y" asks for the student's status and offers further help, and when X is work the response has not supplied it also leaves directive work (Rule 10). A closing courtesy with no identifiable help is neither. A rhetorical question that the assistant answers at once is not Socratic.

## A.5 AI response type (select every value that applies; none of the four applies alone)

Table A.2 gives the meaning of each value.

**Table A.2** Values of AI response type

| Code | Meaning |
|---|---|
| Solve-Provide | Supplies an artefact, answer or fact. |
| Explain | Gives an account of why or how. |
| Correct-Debug | Diagnoses or handles a specific fault. |
| Evaluate | Judges a claim, proposal or piece of work the student put forward. |
| none of the four applies | Only a brief acknowledgement or receipt. |

Solve-Provide and Explain appear in most responses. Correct-Debug and Evaluate are the codes that distinguish responses, so check them with particular care. Asking the student for a contribution belongs to AI delegation and does not by itself establish a response type.

**Rule 12. Evaluate needs a verdict on something the student put forward.** The student's claim, proposal or work can be in the current or recent turns and need not be implemented or visible, for example functions the student names as their own and asks to have checked. A verdict on its correctness or suitability counts however short: "Exactly", "You're right", or "Yes, a good idea" in answer to "Should I ...?". Evaluate is added to the other response types, not chosen instead of them: a verdict that also answers the student's question is Solve-Provide and Evaluate. A judgement of whether the student's actual practice is acceptable counts, even when conditional. General praise does not count, and neither does a judgement of a restatement of what the assistant said, of a method the assignment prescribes, or of an option the assistant introduced, or an answer to a general question that contains no approach of the student's.

Examples:

- A student proposes replacing STAILQ with a plain linked list, and the response confirms that this is possible and weighs it: Evaluate.
- A student states a sorting requirement and asks whether there is a way; "Yes, you can" answers the question: no Evaluate.
- A student proposes a separate test file, and the response answers that it is a good idea: Evaluate, with Solve-Provide.

**Rule 13. Explain needs an account of why or how.** That is a mechanism, a reason or a relationship. A short purpose or reason clause counts ("include stdio.h so that printf is declared"), and so does a short sentence saying what a supplied command or line of code does, such as a line after git rm --cached saying that Git stops tracking the files but leaves them on disk; length and the presence of code are not thresholds. Steps, the output the student should expect to see, a list of what keys or settings mean, comments that only name the next action, and lists of what was changed do not count. Naming the cause of the student's fault is diagnosis (Correct-Debug), not Explain by itself.

Examples:

- A response explains why gcc -o cannot produce a combined object file: Explain. A response that gives the test code only: Solve-Provide. A response that follows a git rm --cached command with a line on what it does to the files: Explain.

**Rule 14. Correct-Debug needs a specific fault in what the student is using or considering.** Judge from what the current response does. The response identifies what goes wrong and under what condition, and diagnoses or handles it. The object can be the student's code, setup, approach or reported failure, commands the assistant gave earlier that the student is now applying, or an option the student raises. The fault need not have occurred and the student need not ask. A requested product that implements a fix diagnosed in earlier turns still counts. A table of common mistakes, a general caution about an API, or a pitfall raised only to contrast with the recommended approach does not count. Technical accuracy does not decide.

Examples:

- A response locates the cause of the memory leak in the student's code: Correct-Debug. A general warning about an API that the student is not using, in an overview, would not count.

**Rule 15. None of the four is for receipts.** A brief acknowledgement or an inventory of received material, with no answer or explanation, is none of the four applies. A substantive summary can still supply an answer, and an answer is Solve-Provide however short, even one sentence.

Examples:

- A response answers the student's thanks with an offer only: none of the four applies.

## A.6 Missing evidence and uncertainty

**Rule 16. Leave a field empty when its evidence is missing.** If the assistant response is truncated, leave both AI fields empty and flag it; code the student fields where the evidence suffices. If an attachment was sent but not preserved, code from the visible material, mark the codes that depend on it as provisional and say so in a note. Do not infer an attachment's content from the assistant's description of it.

**Rule 17. Choose and flag; do not double-select.** When two readings remain plausible, choose the better-supported one and flag the other with the phrase that matters. Multiple selection records co-occurrence, not uncertainty. Notes are short: the phrase or context that decided, a competing reading, or why evidence is unavailable.

---

© 2026 Donghang Wen. This guide is licensed under the Creative Commons Attribution 4.0 International licence (CC BY 4.0, https://creativecommons.org/licenses/by/4.0/). The code in this repository is under the MIT License (LICENSE).
