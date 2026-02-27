# Sprint Planning — Acme Corp Platform Team

**Date:** 2025-07-07
**Attendees:** Alice Chen (Engineering Lead), Bob Martinez (Product Manager), Carol Davis (Designer), David Kim (Backend Engineer), Emma Wilson (QA Engineer), Frank Nguyen (DevOps Engineer), Henry Patel (Frontend Engineer)

---

## Summary

The team held sprint planning for the Acme Corp Platform team. The headline priority is shipping the **new onboarding flow**, which has been on the roadmap for two sprints and has CEO visibility. Secondary items include fixing a login page performance regression affecting enterprise customers, setting up a proper staging environment, and resolving outdated API documentation.

---

## Key Decisions

- The onboarding flow is the top sprint priority, with Carol's designs unblocking David's backend work.
- Frank will set up a staging environment mirroring production by Monday to unblock testing.
- Emma will author the test plan for onboarding once designs are finalised.
- Henry takes ownership of the login page performance regression this sprint.
- API documentation ownership is **unresolved** — Bob to arrange offline.
- Bob will schedule a CEO demo for next Thursday.

---

## Action Items

| # | Task | Owner | Due Date |
|---|------|-------|----------|
| 1 | Finalise mobile screens for onboarding flow | Carol Davis | 2025-07-11 (Friday) |
| 2 | Build onboarding API endpoints | David Kim | 2025-07-18 |
| 3 | Draft onboarding flow test plan | Emma Wilson | 2025-07-18 |
| 4 | Set up production-mirroring staging environment | Frank Nguyen | 2025-07-14 (Monday) |
| 5 | Fix login page performance regression | Henry Patel | 2025-07-18 (end of sprint) |
| 6 | Update outdated API documentation (auth section) | **Unassigned** | 2025-07-18 |
| 7 | Schedule CEO demo for next Thursday | Bob Martinez | 2025-07-07 (today) |

---

## Dependencies

- Task #2 (David — API endpoints) **depends on** Task #1 (Carol — mobile screens by Friday)
- Task #3 (Emma — test plan) **depends on** Task #1 (Carol — mobile screens finalised)
- Testing work broadly **depends on** Task #4 (Frank — staging environment by Monday)
- Task #2 requires a **sync between David Kim and Alice Chen** on auth token scoping

---

## Notes

- CEO has been asking about the onboarding flow — visibility is high.
- Login page regression was reported by **three enterprise customers** — treat as high priority.
- API docs issue has also generated customer complaints; ownership must be resolved promptly.
