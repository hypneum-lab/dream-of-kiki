-- create-paper1-v0p2-mail-drafts.applescript
-- Crée 3 drafts dans Apple Mail (mailbox "Brouillons" — locale FR)
-- pour le workflow submission Paper 1 v0.2 -> PLOS CB.
--
-- Usage:
--   osascript ops/create-paper1-v0p2-mail-drafts.applescript
--
-- Output:
--   3 drafts visibles dans Mail.app > Brouillons (locale française).
--   Aucune envoi automatique : l'utilisateur revoit + envoie manuellement.
--
-- Drafts créés :
--   1. arXiv submission self-notice    (to: c.saillant@gmail.com)
--   2. OSF DOI mint trigger reminder   (to: c.saillant@gmail.com)
--   3. PLOS CB cover letter review     (to: TODO_REVIEWER_EMAIL@example.com)
--
-- Note locale : sur macOS français, le dossier "Drafts" s'appelle
-- "Brouillons". Apple Mail gère automatiquement le routage du
-- message sauvegardé vers le bon mailbox selon la locale système.

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
-- Draft 1 : arXiv submission self-notice
-- ========================================================
set subject1 to "[dreamOfkiki] Paper 1 v0.2 — arXiv submission checklist"
set recipient1 to "c.saillant@gmail.com"
set body1 to "Pré-submission Paper 1 v0.2 — arXiv preprint
Date target : 2026-04-21 ou ASAP

Files à uploader :
- docs/papers/paper1/build/full-draft.tex (567 lines, 22 pages)
- docs/papers/paper1/references.bib

Categories arXiv :
- Primary  : cs.LG  (Machine Learning)
- Cross-list : q-bio.NC (Neurons and Cognition)
- Cross-list : cs.AI

Pre-submit checklist :
- [ ] Authorship byline : \"dreamOfkiki project contributors\"
- [ ] LICENSE MIT (présent au repo root)
- [ ] No AI attribution dans aucun acknowledgement
- [ ] Repo URL : https://github.com/genial-lab/dream-of-kiki
- [ ] OSF link placeholder (sera remplacé post-DOI mint)
- [ ] DR-2 proved (§4.5) — flagger comme nouveau theorem dans abstract

Post-submit action :
- arXiv ID assigned : 2604.XXXXX (à insérer dans :
  - cover-letter PLOS CB
  - Paper 2 cross-cite future
  - docs/milestones/arxiv-submit-log.md (mise à jour)
  - docs/milestones/osf-doi-mint-2026-04-20.md (trigger DOI mint)
)

URL submission : https://arxiv.org/submit"

makeDraftMessage(subject1, recipient1, body1)

-- ========================================================
-- Draft 2 : OSF DOI mint trigger reminder
-- ========================================================
set subject2 to "[dreamOfkiki] Paper 1 v0.2 — OSF DOI mint trigger"
set recipient2 to "c.saillant@gmail.com"
set body2 to "Trigger condition : arXiv ID assigned for Paper 1 v0.2 + Paper 1 v0.2
state frozen.

Action sequence :
1. Login OSF (clement@saillant.cc)
2. Project dreamOfkiki (existing pre-reg there)
3. Amend pre-registration (paste content from
   docs/osf-amendment-bonferroni-cycle3.md)
4. Mint DOI button → DOI assigned 10.17605/OSF.IO/XXXXX
5. Insert DOI in Paper 1 v0.2 §6.1 (preregistration cite)
6. Re-render Paper 1 v0.2 PDF (cd docs/papers/paper1 &&
   pandoc full-draft.md → build/full-draft.tex && pdflatex ×2)
7. Re-upload to arXiv as v2 (replace tex)
8. Update docs/milestones/osf-doi-mint-2026-04-20.md with
   actual DOI

Cross-references :
- Pre-reg amendment file : docs/osf-amendment-bonferroni-cycle3.md
- arXiv submit log : docs/milestones/arxiv-submit-log.md
- OSF mint checklist : docs/milestones/osf-doi-mint-2026-04-20.md"

makeDraftMessage(subject2, recipient2, body2)

-- ========================================================
-- Draft 3 : PLOS CB cover letter review request (collègue)
-- ========================================================
set subject3 to "[Cover letter draft review] Paper 1 v0.2 → PLOS Computational Biology"
set recipient3 to "TODO_REVIEWER_EMAIL@example.com"
set body3 to "Bonjour,

Je m'apprête à soumettre Paper 1 v0.2 du projet dreamOfkiki à
PLOS Computational Biology. Avant submission je voudrais ton avis
sur le cover letter (~575 mots).

Cover letter draft : docs/papers/paper1/cover-letter-plos-cb.md
Paper 1 v0.2 PDF : docs/papers/paper1/build/full-draft.pdf
                  (22 pages, MIT-licensed, repo public)

Repo public : https://github.com/genial-lab/dream-of-kiki

Particulièrement intéressé par ton feedback sur :
- Le 3-bullet PLOS CB scope-fit argument
- La formulation de la contribution principale (DR-2 proved
  + cross-substrate conformance walkthrough §5.6)
- Suggestions de 3 reviewers PLOS CB (placeholders dans le
  cover letter actuel)

Deadline review souhaitée : 48-72h si possible (target submission
2026-04-25).

Merci d'avance.

c.saillant@gmail.com"

makeDraftMessage(subject3, recipient3, body3)

-- Bring Mail to front so user sees the drafts
tell application "Mail"
	activate
end tell

-- Notify completion (Brouillons = Drafts in French locale)
display notification "3 drafts créés dans Brouillons" with title "dreamOfkiki Paper 1 v0.2"

-- Warn if reviewer placeholder remains
if recipient3 contains "TODO_REVIEWER_EMAIL" then
	display notification "Draft 3 : remplacer TODO_REVIEWER_EMAIL avant envoi" with title "dreamOfkiki Paper 1 v0.2"
end if

return "3 drafts created in Mail.app > Brouillons. Review and send manually."
