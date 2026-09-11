# RABBiT preview — restrained blue revision

final result: passed

Reviewed 2026-09-08. Scope: the isolated design preview; this is not approval to promote it to the main site.

## Findings and comparison history

- [P1, resolved] The previous dark blue-green hero/footer and colored reading sections departed from the user's intended restrained blue identity. Removed these fills and sea-green accents. Restored white, graphite text, and blue related to the comparison legend. The user’s explicit correction supersedes the previous palette revision.
- The current side-by-side review found no remaining actionable P0/P1/P2 issues. The subtle hero panel, blue heading emphasis, masthead rule, and typographic refinements are intentional responses to the request for character. They are not claims of exact pixel matching to the earlier mockup.

## Evidence

- Visual reference: `../../design-review/mockup.png`, 932 × 1688 px, supplemented by the user's 2026-09-08 direction.
- Implementation: `../../design-review/reference.png`, 917 × 1824 px from a 932 × 1688 CSS viewport at device scale 1; the 15px difference is the browser scrollbar.
- Full comparison: `../../design-review/comparison.png`, 1888 × 2050 px. Reference and implementation are presented at equal 932px widths, preserving aspect ratio; this gives the implementation a small 1.016× normalization. The real page includes additional scientific qualifications and a full-size figure link.
- Readable full-page checks: `../../design-review/desktop.png` (1425 × 2142 px, CSS 1440 × 1000), `mobile.png` (375 × 2583 px, CSS 390 × 844), and `small-mobile.png` (305 × 2787 px, CSS 320 × 720). All captured in the resting light state at scale 1, with reduced motion. Desktop and mobile screenshots were opened and inspected. No extra focused comparison was needed: the side-by-side view plus readable desktop capture exposed all changed typography, surfaces, borders, and figure framing.

## Five fidelity surfaces

- Typography: retains the self-hosted Inter family and original wordmark. Tighter hero leading, 600-weight headings, and smaller tracked labels add hierarchy. The headline remains three lines on desktop; mobile wraps naturally. No new font family.
- Layout: retains the selected two-column hero, paired use cases, comparison row, and compact footer. The use-case text aligns with the section heading. Mobile stacks cleanly; the hero panel has room for its caption.
- Color: white page (#ffffff), graphite ink (#202936), blue accent (#2c5885), and blue-gray rules (#dce4ed). Only the hero figure uses a faint cool surface (#f5f8fc). Text contrast: body 8.59:1, muted text 5.12:1, caption 4.80:1, blue links/button 7.39:1.
- Imagery: original scientific image files and data are unchanged. Transparent hero render sits on the faint panel; the comparison map retains white behind its original legend. No clipping, stretching, or visible loading failures.
- Copy: scientific content and qualifications are unchanged. The only HTML content markup change is a span around “brain activity.” to give it blue emphasis.

## Verification and limits

Existing browser checks passed at 932, 1440, 390, and 320px: fonts and images loaded, no horizontal overflow, no console/page/request errors, keyboard skip link works, Results anchor works, and local destinations return successful responses. Results: `../../design-review/check-results.json`. `git diff --check` passes. Remote paper/code availability and the existing inference demo were not retested in this visual-only revision.

## Implementation checklist

- [x] Restore the restrained blue identity on white.
- [x] Add character through typography, alignment, and limited figure framing.
- [x] Compare against the original mockup and inspect responsive views.
- [x] Record the user's correction in the review notes and plan.

No further changes are required for local visual review. Promotion to the main landing page remains pending user review.
