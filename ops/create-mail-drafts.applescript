-- Create 4 Apple Mail drafts for dreamOfkiki T-Col outreach
-- Run: osascript ops/create-mail-drafts.applescript
-- Drafts appear in Mail.app > Drafts folder, ready for review + send
--
-- Required env vars (falls back to placeholders if unset):
--   DREAMOFKIKI_HUTH_EMAIL
--   DREAMOFKIKI_NORMAN_EMAIL
--   DREAMOFKIKI_GALLANT_EMAIL
-- Example usage from shell:
--   DREAMOFKIKI_HUTH_EMAIL=alex@cs.utexas.edu \
--   DREAMOFKIKI_NORMAN_EMAIL=knorman@princeton.edu \
--   DREAMOFKIKI_GALLANT_EMAIL=gallant@berkeley.edu \
--   osascript ops/create-mail-drafts.applescript

on makeDraftMessage(emailSubject, emailRecipient, emailBody)
	tell application "Mail"
		set newMessage to make new outgoing message with properties {subject:emailSubject, content:emailBody, visible:false}
		tell newMessage
			make new to recipient with properties {address:emailRecipient}
		end tell
		save newMessage
	end tell
end makeDraftMessage

-- ========================================================
-- Email 1: Huth Lab (UT Austin) — fMRI collaboration
-- ========================================================
set subject1 to "dreamOfkiki - fMRI RSA collaboration invitation"
set recipient1 to do shell script "echo ${DREAMOFKIKI_HUTH_EMAIL:-PLACEHOLDER_HUTH@example.com}"
set body1 to "Dear Dr. Huth,

I am Clement Saillant, an independent embedded/ML researcher at L'Electron Rare (France). I am reaching out regarding potential collaboration on fMRI representational similarity analysis (RSA) work.

I am running a formal research program called dreamOfkiki that axiomatizes dream-based knowledge consolidation in artificial cognitive systems. The framework produces two papers: a theoretical one (formal framework with executable axioms, cycle 1 target Nature Human Behaviour) and an empirical ablation on a hierarchical linguistic model called kiki-oniric (cycle 1 target NeurIPS/ICML/TMLR).

One of our pre-registered hypotheses (H3, OSF DOI pending) predicts that representational alignment between kiki embeddings and fMRI activity in language ROIs (STG, IFG, AG) increases monotonically across three consolidation profiles. We have locked Studyforrest as a fallback dataset (feasibility verified at https://github.com/electron-rare/dream-of-kiki/blob/main/docs/feasibility/studyforrest-rsa-note.md), but your Narratives dataset work would strengthen the alignment substantially.

I would value the opportunity to discuss a potential collaboration in exchange for courtesy co-authorship on the framework paper (cycle 1 submission around June 2026). Time estimate on your side: approximately 2-3 hours of review, plus data access coordination if we agree on the fit.

Repository (code + specs): https://github.com/electron-rare/dream-of-kiki

Happy to send the formal framework draft, the pre-registration package, or answer questions. Would you be available for a brief email exchange or call?

Best regards,
Clement Saillant
L'Electron Rare, France
clement@saillant.cc"

makeDraftMessage(subject1, recipient1, body1)

-- ========================================================
-- Email 2: Norman Lab (Princeton) — fMRI episodic memory
-- ========================================================
set subject2 to "dreamOfkiki - fMRI consolidation collaboration"
set recipient2 to do shell script "echo ${DREAMOFKIKI_NORMAN_EMAIL:-PLACEHOLDER_NORMAN@example.com}"
set body2 to "Dear Dr. Norman,

I am Clement Saillant, an independent embedded/ML researcher at L'Electron Rare (France). I am reaching out regarding potential collaboration on episodic memory consolidation work bridging cognitive neuroscience and AI systems.

I am running a formal research program called dreamOfkiki that axiomatizes dream-based knowledge consolidation in artificial cognitive systems. The theoretical framework builds on complementary learning systems (McClelland 1995), synaptic homeostasis (Tononi SHY), predictive processing (Friston FEP), and creative dreaming (Hobson/Solms) as algebraically composable operations. We produce two cycle 1 papers: a framework paper targeting Nature Human Behaviour, and an empirical ablation targeting NeurIPS.

Your expertise on episodic memory and its consolidation dynamics would substantially strengthen the cognitive validity framing of our ablation, particularly for our H3 hypothesis (monotonic alignment across three consolidation profiles). We have locked Studyforrest as fallback fMRI data (feasibility verified) but would welcome discussion of complementary episodic memory datasets.

In exchange for collaboration (data access guidance or direct co-analysis), we offer courtesy co-authorship on the framework paper. Time estimate: approximately 2-3 hours plus data coordination.

Repository (code + specs): https://github.com/electron-rare/dream-of-kiki

Happy to send the formal framework draft or the pre-registration package. Would you be available for a brief email exchange?

Best regards,
Clement Saillant
L'Electron Rare, France
clement@saillant.cc"

makeDraftMessage(subject2, recipient2, body2)

-- ========================================================
-- Email 3: Gallant Lab (UC Berkeley) — natural stimuli fMRI
-- ========================================================
set subject3 to "dreamOfkiki - naturalistic fMRI RSA inquiry"
set recipient3 to do shell script "echo ${DREAMOFKIKI_GALLANT_EMAIL:-PLACEHOLDER_GALLANT@example.com}"
set body3 to "Dear Dr. Gallant,

I am Clement Saillant, an independent embedded/ML researcher at L'Electron Rare (France). I am reaching out regarding potential collaboration on naturalistic fMRI alignment with artificial linguistic models.

I am running a formal research program called dreamOfkiki that axiomatizes dream-based knowledge consolidation in artificial cognitive systems. Our framework paper (cycle 1 target Nature Human Behaviour) defines executable axioms DR-0..DR-4 over four primitive operations (replay, downscale, restructure, recombine) as an algebraic structure. Our empirical ablation (cycle 1 target NeurIPS) tests three consolidation profiles against a baseline on a hierarchical linguistic model.

Your lab's work on naturalistic stimulus-driven BOLD responses and semantic tiling is highly relevant for our pre-registered H3 hypothesis: representational alignment between kiki embeddings and fMRI should increase monotonically across profiles. We have Studyforrest locked as fallback, but naturalistic semantic data from your group would be an excellent complement.

In exchange for collaboration, we offer courtesy co-authorship on the framework paper (cycle 1 submission June 2026). Time estimate approximately 2-3 hours plus data coordination.

Repository (code + specs): https://github.com/electron-rare/dream-of-kiki

Happy to send the formal framework draft or our pre-registration package. Would you be available for a brief email exchange to explore fit?

Best regards,
Clement Saillant
L'Electron Rare, France
clement@saillant.cc"

makeDraftMessage(subject3, recipient3, body3)

-- ========================================================
-- Email 4: Formal reviewer template (recipient placeholder)
-- ========================================================
set subject4 to "dreamOfkiki - DR-2 compositionality proof review request"
set recipient4 to "REVIEWER_EMAIL_TO_FILL@example.com"
set body4 to "Dear [REVIEWER NAME],

[CONTEXT HOOK - adjust per candidate: where we met, shared work, mutual contact]

I am finishing up a research program called dreamOfkiki - a formal framework axiomatizing dream-based knowledge consolidation in artificial cognitive systems. The framework defines four primitive operations (replay, downscale, restructure, recombine) as a monoid-like algebraic structure over cognitive state transitions, with a compositionality axiom (DR-2) that is central to the framework's substrate-agnosticism claim (DR-3).

I am reaching out because I would value a formal reviewer's eye on the DR-2 compositionality proof before we submit the paper to Nature Human Behaviour. The proof sketch is in place (closure, budget additivity, functional composition, associativity), but I would like an external pair of eyes on the case analysis for non-commutative operation pairs, in particular recombine after downscale versus downscale after recombine.

Time estimate: 2-3 hours of your time.

In return: courtesy co-authorship on the paper + credit in CONTRIBUTORS.md for the research program. The paper is scheduled for submission around S20 (June 2026 in our calendar).

Materials I can share: draft proof (approximately 3 pages), framework spec section 6 (axioms DR-0..DR-4), context on the monoid construction.

Repository (code + specs): https://github.com/electron-rare/dream-of-kiki

Would you be available around S6-S8 (circulation and review window) ? Happy to chat via email or a short call beforehand if that is easier.

Best regards,
Clement Saillant
L'Electron Rare, France
clement@saillant.cc"

makeDraftMessage(subject4, recipient4, body4)

-- Bring Mail to front so user sees the drafts
tell application "Mail"
	activate
end tell

-- Warn if any placeholder remained
if recipient1 contains "PLACEHOLDER" or recipient2 contains "PLACEHOLDER" or recipient3 contains "PLACEHOLDER" then
	display notification "Set DREAMOFKIKI_{HUTH,NORMAN,GALLANT}_EMAIL env vars then re-run" with title "dreamOfkiki Mail drafts"
end if

return "4 drafts created in Mail.app > Drafts. Review and send manually."
