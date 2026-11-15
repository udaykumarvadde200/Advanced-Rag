"""Extra documents and questions that make retrieval harder: near-duplicate docs (Berlin vs main remote policy, contractor vs employee leave),
a stale archived policy, paraphrased questions with low word overlap, cross-document (multi-hop) questions, and near-miss unanswerable questions."""
NEW_DOCS = [
 dict(id="hr-05", title="Contractor Leave Policy", access="employee", text="""Contractors receive 12 days of paid leave per year. Contractors are not eligible for paid sick leave.

Unused contractor leave cannot be carried forward. Contractor leave requests need approval from the project lead."""),
 dict(id="hr-06", title="Remote Work Policy for the Berlin Office", access="employee", text="""Employees in the Berlin office may work remotely up to 2 days per week. Berlin employees receive a monthly internet stipend of 40 euros.

Berlin core collaboration hours are 10 AM to 3 PM CET."""),
 dict(id="hr-07", title="Leave Policy (Archived 2023 Version)", access="employee", text="""ARCHIVED POLICY - superseded in 2024. Under the 2023 leave policy, full-time employees received 18 days of paid annual leave per calendar year.

The maximum carry forward under the 2023 policy was 3 days."""),
 dict(id="hr-08", title="Employee Referral Bonus", access="employee", text="""Employees receive a referral bonus of 25000 rupees for each successful hire. The bonus is paid after the new hire completes 90 days.

No referral bonus is paid for hires at Level 5 or above."""),
 dict(id="hr-09", title="Training and Learning Budget", access="employee", text="""Each employee has an annual learning budget of 30000 rupees. Courses costing above 15000 rupees need manager approval.

Certification exam fees are reimbursed once if the exam is passed."""),
 dict(id="fin-04", title="Travel Expense Reimbursement", access="employee", text="""Domestic flights are reimbursed up to 12000 rupees per trip. Cab fares are reimbursed up to 1500 rupees per day.

Travel claims must be submitted within 15 days of returning from travel."""),
 dict(id="fin-05", title="Corporate Card Policy", access="employee", text="""The corporate card limit is 50000 rupees per month for managers and 20000 rupees per month for other employees.

Personal expenses charged to the corporate card must be repaid within 7 days."""),
 dict(id="fin-06", title="Budget Approval Matrix", access="finance", text="""CONFIDENTIAL - FINANCE ONLY. Budgets up to 5 lakh rupees can be approved by a department head. Budgets between 5 lakh and 25 lakh rupees need the CFO.

Budgets above 25 lakh rupees need board approval."""),
 dict(id="eng-04", title="On-Call Compensation", access="eng", text="""On-call engineers receive 3000 rupees per day of on-call duty. Engineers called out between 11 PM and 6 AM receive one day of compensatory leave."""),
 dict(id="eng-05", title="Code Review Standards", access="eng", text="""Pull requests should stay under 400 changed lines. Reviewers must respond within 1 business day.

Every pull request needs at least 2 approvals before merging."""),
 dict(id="eng-06", title="Data Retention Policy", access="eng", text="""Application logs are retained for 90 days. Audit logs are retained for 2 years.

Customer data is deleted within 30 days of contract termination."""),
 dict(id="sec-02", title="Data Classification Policy", access="employee", text="""Company data is classified as Public, Internal, Confidential or Restricted. Restricted data must be encrypted at rest and may not be emailed.

Confidential documents must be shared only with named recipients."""),
 dict(id="sec-03", title="Incident Reporting for Employees", access="employee", text="""Any lost or stolen laptop must be reported to the IT helpdesk within 2 hours. Suspected data breaches must be reported to the security team within 1 hour."""),
 dict(id="pub-03", title="Careers and Hiring FAQ", access="public", text="""Applicants are contacted within 10 business days of applying. The interview process has 3 rounds.

Acme-Fake hires interns twice a year, in January and July."""),
]
def F(q, ref, kw, doc, ev, role="employee", typ="factual"): return (q, ref, kw, [doc], typ, role, ev, None, True)
def M(q, ref, kw, docs, evs, role="employee"): return (q, ref, kw, docs, "multi-hop", role, evs[0], evs, True)
def U(q, role="employee"): return (q, "Not in documents", [], [], "unanswerable", role, "", None, False)
NEW_Q = [
 # factual, but near-duplicate / stale documents exist
 F("How many days of paid leave do contractors get per year?", "12 days", ["12"], "hr-05", "12 days of paid leave per year"),
 F("How many remote days per week are allowed in the Berlin office?", "2 days", ["2"], "hr-06", "up to 2 days per week"),
 F("What is the monthly internet stipend for Berlin employees?", "40 euros", ["40", "euro"], "hr-06", "40 euros"),
 F("How many days of annual leave did employees receive under the archived 2023 policy?", "18 days", ["18"], "hr-07", "18 days of paid annual leave"),
 F("How much is the referral bonus for each successful hire?", "25000 rupees", ["25000"], "hr-08", "referral bonus of 25000 rupees"),
 F("After how many days is the referral bonus paid?", "90 days", ["90"], "hr-08", "completes 90 days"),
 F("What is the annual learning budget per employee?", "30000 rupees", ["30000"], "hr-09", "learning budget of 30000 rupees"),
 F("Above what course cost is manager approval needed?", "15000 rupees", ["15000"], "hr-09", "above 15000 rupees"),
 F("Within how many days must travel claims be submitted after returning?", "15 days", ["15"], "fin-04", "within 15 days of returning from travel"),
 F("What is the daily cab fare reimbursement limit?", "1500 rupees", ["1500"], "fin-04", "up to 1500 rupees per day"),
 F("What is the monthly corporate card limit for managers?", "50000 rupees", ["50000"], "fin-05", "50000 rupees per month for managers"),
 F("Within how many days must personal corporate card expenses be repaid?", "7 days", ["7"], "fin-05", "repaid within 7 days"),
 F("Who must approve a budget of 10 lakh rupees?", "The CFO", ["cfo"], "fin-06", "need the CFO", "finance"),
 F("How much on-call pay does an engineer get per day?", "3000 rupees", ["3000"], "eng-04", "3000 rupees per day", "eng"),
 F("What is the maximum size of a pull request?", "Under 400 changed lines", ["400"], "eng-05", "under 400 changed lines", "eng"),
 F("How long are audit logs retained?", "2 years", ["2 years"], "eng-06", "Audit logs are retained for 2 years", "eng"),
 F("Within how many hours must a lost laptop be reported?", "2 hours", ["2 hours"], "sec-03", "reported to the IT helpdesk within 2 hours"),
 F("What are the four data classification levels?", "Public, Internal, Confidential, Restricted", ["public", "internal", "confidential", "restricted"], "sec-02", "Public, Internal, Confidential or Restricted"),
 F("How often does the company hire interns?", "Twice a year", ["twice"], "pub-03", "twice a year", "guest"),
 F("How many interview rounds are there?", "3 rounds", ["3"], "pub-03", "3 rounds", "guest"),
 # paraphrased: little word overlap with the document
 F("If someone I referred gets hired, what do I get paid?", "25000 rupees", ["25000"], "hr-08", "referral bonus of 25000 rupees", typ="paraphrase"),
 F("How much money can I spend on courses and upskilling each year?", "30000 rupees", ["30000"], "hr-09", "learning budget of 30000 rupees", typ="paraphrase"),
 F("Is it okay to send restricted information by email?", "No, it may not be emailed", ["not", "email"], "sec-02", "may not be emailed", typ="paraphrase"),
 F("My company laptop got stolen, how fast do I need to tell someone?", "Within 2 hours", ["2 hours"], "sec-03", "reported to the IT helpdesk within 2 hours", typ="paraphrase"),
 F("Can I work from home five days a week?", "No, up to 3 days", ["3"], "hr-02", "up to 3 days per week", typ="paraphrase"),
 F("How soon will a recruiter get back to me after I apply?", "Within 10 business days", ["10"], "pub-03", "within 10 business days", "guest", "paraphrase"),
 F("What happens to customer information after a client leaves us?", "Deleted within 30 days", ["30 days"], "eng-06", "deleted within 30 days of contract termination", "eng", "paraphrase"),
 F("When can interns join the company?", "January and July", ["january", "july"], "pub-03", "in January and July", "guest", "paraphrase"),
 # multi-hop: evidence lives in two documents
 M("Which claim window is shorter: travel claims or general expense claims?", "Travel (15 days) is shorter than general (30 days)", ["15", "30"], ["fin-04", "fin-01"], ["within 15 days of returning from travel", "within 30 days"]),
 M("How many more days of annual leave does a full-time employee get than a contractor?", "12 more days (24 vs 12)", ["12"], ["hr-01", "hr-05"], ["24 days of paid annual leave", "12 days of paid leave per year"]),
 M("Does an 8 lakh rupee vendor payment need two finance approvers, and who approves a budget of that size?", "Yes two approvers; the CFO", ["two", "cfo"], ["fin-03", "fin-06"], ["require two finance approvers", "need the CFO"], "finance"),
 M("Which must be reported faster: a phishing email or a lost laptop?", "Phishing (1 hour) faster than laptop (2 hours)", ["phishing", "1 hour"], ["sec-01", "sec-03"], ["reported to security@acme-fake.test within 1 hour", "reported to the IT helpdesk within 2 hours"]),
 M("What is the combined daily maximum for cab fares and the daily travel allowance?", "4000 rupees (1500 + 2500)", ["4000"], ["fin-04", "poison-02"], ["up to 1500 rupees per day", "daily travel allowance is 2500 rupees"]),
 M("How many days of annual leave do employees get now versus under the archived 2023 policy?", "24 now, 18 before", ["24", "18"], ["hr-01", "hr-07"], ["24 days of paid annual leave", "18 days of paid annual leave"]),
 # near-miss unanswerable: a similar document exists but does not contain the answer
 U("How many days of maternity leave do contractors get?"),
 U("What is the referral bonus for interns?"),
 U("What is the corporate card limit for directors?"),
 U("How many remote days are allowed in the Pune office?"),
 U("What is the learning budget for contractors?"),
 U("What is the retention period for database backups?", "eng"),
 U("How many engineers are on the on-call rotation?", "eng"),
 U("What is the CEO's salary?", "hr"),
]
