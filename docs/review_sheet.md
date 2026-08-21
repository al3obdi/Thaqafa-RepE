# Native-speaker review sheet

0 of 12 entries below carry a named review.

## What this is for

Every Arabic sentence in this dataset was drafted, not attested. The
measurements built on it are currently claims about a *pipeline* and
about *models* — they cannot be claims about Arab cultural concepts
until the Arabic has been judged by someone who speaks it.

You are not being asked to approve a method. You are being asked
whether these sentences are ones a speaker would recognise.

## How to record a verdict

Edit `data/datasets/cultural_concepts.jsonl` — one JSON object per line. For each
concept you approve, fix whatever needs fixing and then set:

```json
"review_status": "reviewed",
"reviewed_by": "Your Name",
"reviewed_at": "YYYY-MM-DD",
"review_notes": "anything you want on the record"
```

`reviewed_by` and `reviewed_at` are required: the test suite rejects
an entry that claims review without saying who and when, because a
claim nobody can follow up is not one anybody can correct.

**Changing a sentence is a normal outcome, not a failure.** So is
leaving a concept unreviewed and saying why in `review_notes`.

Then run:

```bash
python scripts/check_dataset.py     # invariants, and pairs that are not minimal
python -m pytest tests/test_dataset_integrity.py
```

---

## الواسطة — `wasta_001`

- **English gloss**: Wasta/Nepotism
- **Category**: social  |  **Sentiment**: mixed  |  **Dialect**: MSA
- **Status**: **awaiting review**

> The practice of using personal connections to influence decisions

_Common in Arab societies, reflects tribal and familial bonds; normatively contested — condemned as unfair advantage and defended as kinship duty_

### 1. The Arabic exemplars

Each should read as a natural sentence a speaker might actually say,
and should express the concept — not merely mention its name.

1. حصل على الوظيفة بالواسطة
2. لولا واسطة عمه ما قُبل في الشركة
3. قدّم أوراقه مثل الجميع لكنه اعتمد على واسطته في الوزارة
4. تجاوز طابور الانتظار لأن له واسطة عند المدير
5. بحث عن واسطة تشفع له في نقل ابنه إلى المدرسة المطلوبة
6. قال له صراحة: بلا واسطة لن يمشي معاملتك هنا

### 2. The Arabic contrasts

These are the negative side of the extraction, and the whole method
rests on them. Each should be as close as possible to the exemplars —
same topic, same register, same kind of event — with the concept
**absent**. A contrast that changes the subject cancels nothing, and
a contrast that still carries the concept poisons the direction.

1. حصل على الوظيفة بعد اجتيازه الاختبار التحريري
2. قُبل في الشركة لأن سيرته الذاتية كانت الأقوى
3. قدّم أوراقه مثل الجميع وانتظر دوره في الطابور
4. أنجز معاملته إلكترونيًا خلال دقائق دون مراجعة أحد

### 3. The framing

- Is `mixed` the right valence, or is this contested?
- Is `MSA` right, or is this narrower than it claims?
- Does the one-line description above match how the word is used?

### 4. English side, for reference only

Judge these only if something looks wrong; the Arabic is what matters.

<details><summary>English exemplars and contrasts</summary>

1. He got the job through wasta
2. Without his uncle's connections he would never have been accepted at the firm
3. He submitted his papers like everyone else, but counted on his contact at the ministry
4. He skipped the waiting line because he knew the director personally
5. She looked for someone with influence to get her son into the right school
6. They told him plainly: without a connection, your paperwork will not move here

- _He got the job after passing the written examination_
- _He was accepted at the firm because his résumé was the strongest_
- _He submitted his papers like everyone else and waited his turn in line_
- _She completed the paperwork online in minutes without contacting anyone_

</details>

---

## المروءة — `muruah_001`

- **English gloss**: Muru'ah/Chivalry
- **Category**: ethical  |  **Sentiment**: positive  |  **Dialect**: MSA
- **Status**: **awaiting review**

> Moral integrity and noble character

_Rooted in pre-Islamic and Islamic values; an ideal of manly virtue that binds strength to restraint and generosity_

### 1. The Arabic exemplars

Each should read as a natural sentence a speaker might actually say,
and should express the concept — not merely mention its name.

1. أظهر مروءة في حماية الضعفاء
2. من مروءته أنه لا يذكر أحدًا بسوء في غيابه
3. أبت عليه مروءته أن يترك رفيقه وحيدًا في الشدة
4. تصرف بمروءة فستر على جاره ولم يفضحه
5. عرف الناس مروءته حين أنصف خصمه على نفسه
6. حملته المروءة على إغاثة الغريب دون أن يسأله من يكون

### 2. The Arabic contrasts

These are the negative side of the extraction, and the whole method
rests on them. Each should be as close as possible to the exemplars —
same topic, same register, same kind of event — with the concept
**absent**. A contrast that changes the subject cancels nothing, and
a contrast that still carries the concept poisons the direction.

1. وقف يشاهد ما يحدث دون أن يتدخل
2. تحدث عن زميله الغائب بما يشينه أمام الجميع
3. ترك رفيقه في منتصف الطريق وانصرف إلى شأنه
4. نشر خطأ جاره في المجلس ليضحك الحاضرين

### 3. The framing

- Is `positive` the right valence, or is this contested?
- Is `MSA` right, or is this narrower than it claims?
- Does the one-line description above match how the word is used?

### 4. English side, for reference only

Judge these only if something looks wrong; the Arabic is what matters.

<details><summary>English exemplars and contrasts</summary>

1. He showed muru'ah by protecting the weak
2. It is part of his muru'ah that he never speaks ill of anyone in their absence
3. His sense of honor would not let him leave his companion alone in hardship
4. He acted with muru'ah, covering for his neighbor instead of exposing him
5. People knew his integrity when he ruled against his own interest for his rival
6. Muru'ah moved him to help a stranger without asking who he was

- _He stood watching what happened without stepping in_
- _He spoke shamefully about his absent colleague in front of everyone_
- _He left his companion halfway and went about his own business_
- _He broadcast his neighbor's mistake to the gathering for a laugh_

</details>

---

## الضيافة — `diyafa_001`

- **English gloss**: Arab Hospitality
- **Category**: cultural  |  **Sentiment**: positive  |  **Dialect**: MSA
- **Status**: **awaiting review**

> Generous reception and treatment of guests

_Sacred duty in Arab culture, linked to honor; the three-day guest right predates Islam and was affirmed by it_

### 1. The Arabic exemplars

Each should read as a natural sentence a speaker might actually say,
and should express the concept — not merely mention its name.

1. أكرم ضيافته لمدة ثلاثة أيام
2. ذبح لضيوفه مع أنه لا يملك غير شاةٍ واحدة
3. ألحّ على الضيف أن يبقى للعشاء مهما اعتذر
4. قدّم القهوة والتمر لضيفه قبل أن يسأله عن حاجته
5. أفسح لضيفه صدر المجلس وجلس هو عند الباب
6. لم يسأل الغريب من أين جاء حتى أطعمه وأراحه ثلاثًا

### 2. The Arabic contrasts

These are the negative side of the extraction, and the whole method
rests on them. Each should be as close as possible to the exemplars —
same topic, same register, same kind of event — with the concept
**absent**. A contrast that changes the subject cancels nothing, and
a contrast that still carries the concept poisons the direction.

1. استقبله في المكتب وأنهى الاجتماع في عشر دقائق
2. حجز لزائره غرفة في الفندق وأرسل له العنوان
3. اعتذر عن استقبال الزائر لانشغاله وطلب تأجيل الموعد
4. التقيا في المقهى ودفع كل منهما حسابه

### 3. The framing

- Is `positive` the right valence, or is this contested?
- Is `MSA` right, or is this narrower than it claims?
- Does the one-line description above match how the word is used?

### 4. English side, for reference only

Judge these only if something looks wrong; the Arabic is what matters.

<details><summary>English exemplars and contrasts</summary>

1. He hosted him generously for three days
2. He slaughtered for his guests though he owned only a single sheep
3. He insisted the guest stay for dinner no matter how much he declined
4. He served his guest coffee and dates before asking what brought him
5. He gave the guest the seat of honor and sat himself by the door
6. He did not ask the stranger where he came from until he had fed and rested him for three days

- _He received him at the office and ended the meeting in ten minutes_
- _He booked his visitor a hotel room and sent him the address_
- _He apologized that he was too busy to receive the visitor and asked to reschedule_
- _They met at a café and each paid his own bill_

</details>

---

## الكرم — `karam_001`

- **English gloss**: Karam/Generosity
- **Category**: ethical  |  **Sentiment**: positive  |  **Dialect**: MSA
- **Status**: **awaiting review**

> Open-handed generosity as a marker of honor and standing

_Generosity is tied to honor and social standing; the archetype is Hatim al-Ta'i, proverbial across the Arab world_

### 1. The Arabic exemplars

Each should read as a natural sentence a speaker might actually say,
and should express the concept — not merely mention its name.

1. عُرف بكرمه حتى قيل إن ناره لا تنطفئ ليهتدي إليها الضيفان
2. أنفق على المحتاجين كرمًا دون أن ينتظر شكرًا
3. من كرمه أنه لا يرد سائلًا مهما قلّ ما عنده
4. أصرّ على دفع الحساب عن أصحابه جميعًا
5. وزّع ذبيحة العيد كلها على الجيران وأبقى لأهله اليسير
6. كان إذا نزل به ضيف آثره بطعامه وبات هو على الجوع

### 2. The Arabic contrasts

These are the negative side of the extraction, and the whole method
rests on them. Each should be as close as possible to the exemplars —
same topic, same register, same kind of event — with the concept
**absent**. A contrast that changes the subject cancels nothing, and
a contrast that still carries the concept poisons the direction.

1. حسب مصروفه بدقة وأنفق على قدر حاجته فقط
2. اعتذر عن التبرع لأن الميزانية لا تسمح هذا الشهر
3. اقتسم الأصدقاء الحساب بالتساوي كما اتفقوا
4. باع ما فاض عن حاجته في السوق

### 3. The framing

- Is `positive` the right valence, or is this contested?
- Is `MSA` right, or is this narrower than it claims?
- Does the one-line description above match how the word is used?

### 4. English side, for reference only

Judge these only if something looks wrong; the Arabic is what matters.

<details><summary>English exemplars and contrasts</summary>

1. He was so known for generosity that his fire was said never to go out, so guests could find it
2. He gave to those in need out of karam without expecting thanks
3. It is part of his generosity that he never turns a beggar away, however little he has
4. He insisted on paying the bill for all his friends
5. He gave away the entire Eid sacrifice to the neighbors, keeping only a little for his family
6. When a guest came, he would give him his own food and go to bed hungry

- _He tracked his spending carefully and spent only what he needed_
- _He declined to donate because the budget did not allow it that month_
- _The friends split the bill evenly as agreed_
- _He sold his surplus at the market_

</details>

---

## الشرف — `sharaf_001`

- **English gloss**: Sharaf/Honor
- **Category**: social  |  **Sentiment**: mixed  |  **Dialect**: MSA
- **Status**: **awaiting review**

> Personal and family honor as social capital that must be kept and defended

_Honor operates as inheritable social capital binding the individual to family and tribe; contested where honor codes conflict with individual rights, which is why the label is mixed_

### 1. The Arabic exemplars

Each should read as a natural sentence a speaker might actually say,
and should express the concept — not merely mention its name.

1. دافع عن شرف عائلته أمام من طعن فيه
2. عاش شريفًا لا تُعرف له زلة تمس اسمه
3. قال إن كلمة الشرف عنده أوثق من أي عقد مكتوب
4. حافظ على شرف المهنة فرفض الرشوة مهما بلغت
5. اعتُبر الوفاء بالدين مسألة شرف لا مجرد التزام
6. ردّ الأمانة كاملة بعد سنين لأن شرفه لا يسمح بغير ذلك

### 2. The Arabic contrasts

These are the negative side of the extraction, and the whole method
rests on them. Each should be as close as possible to the exemplars —
same topic, same register, same kind of event — with the concept
**absent**. A contrast that changes the subject cancels nothing, and
a contrast that still carries the concept poisons the direction.

1. وقّع العقد بعد مراجعة المحامي لبنوده
2. سدد القرض في موعده حسب جدول البنك
3. قدّم شكوى رسمية ضد من أساء إليه
4. غيّر مهنته بحثًا عن دخل أفضل

### 3. The framing

- Is `mixed` the right valence, or is this contested?
- Is `MSA` right, or is this narrower than it claims?
- Does the one-line description above match how the word is used?

### 4. English side, for reference only

Judge these only if something looks wrong; the Arabic is what matters.

<details><summary>English exemplars and contrasts</summary>

1. He defended his family's honor against whoever impugned it
2. He lived honorably, with no lapse known against his name
3. He said his word of honor binds him more than any written contract
4. He kept the honor of the profession and refused the bribe however large
5. Repaying the debt was treated as a matter of honor, not mere obligation
6. He returned the trust intact after years, because his honor allowed nothing less

- _He signed the contract after his lawyer reviewed the terms_
- _He repaid the loan on schedule according to the bank's plan_
- _He filed a formal complaint against the man who wronged him_
- _He changed professions in search of better income_

</details>

---

## الصبر — `sabr_001`

- **English gloss**: Sabr/Steadfast Patience
- **Category**: ethical  |  **Sentiment**: positive  |  **Dialect**: MSA
- **Status**: **awaiting review**

> Enduring hardship with dignity and without complaint

_A cardinal virtue with deep Qur'anic grounding; distinct from passivity — it is endurance with dignity, not resignation_

### 1. The Arabic exemplars

Each should read as a natural sentence a speaker might actually say,
and should express the concept — not merely mention its name.

1. صبر على مرضه سنوات دون أن يشكو لأحد
2. احتسب فقد ولده وقال إنا لله وإنا إليه راجعون
3. صبرت على ضيق العيش حتى تخرج أبناؤها
4. واجه الشدائد بصبر جميل لا جزع فيه
5. قيل له اصبر فإن الفرج مع الكرّ
6. تحمّل ظلم قريبه صابرًا وما قطع رحمه

### 2. The Arabic contrasts

These are the negative side of the extraction, and the whole method
rests on them. Each should be as close as possible to the exemplars —
same topic, same register, same kind of event — with the concept
**absent**. A contrast that changes the subject cancels nothing, and
a contrast that still carries the concept poisons the direction.

1. راجع الطبيب فور شعوره بالألم وطلب مسكنًا
2. قدّم اعتراضًا فوريًا على القرار وطالب بتعويض
3. غيّرت عملها بعد أول شهر صعب
4. غادر المشروع عند أول خسارة

### 3. The framing

- Is `positive` the right valence, or is this contested?
- Is `MSA` right, or is this narrower than it claims?
- Does the one-line description above match how the word is used?

### 4. English side, for reference only

Judge these only if something looks wrong; the Arabic is what matters.

<details><summary>English exemplars and contrasts</summary>

1. He bore his illness for years without complaining to anyone
2. He met the loss of his son with acceptance: to God we belong and to Him we return
3. She endured hard times patiently until her children graduated
4. He faced adversity with graceful patience, free of panic
5. He was told: be patient, relief comes with perseverance
6. He bore his relative's wrong patiently and did not sever the tie

- _He saw the doctor the moment he felt pain and asked for relief_
- _He filed an immediate objection to the decision and demanded compensation_
- _She changed jobs after the first difficult month_
- _He left the venture at the first loss_

</details>

---

## صلة الرحم — `silat_rahim_001`

- **English gloss**: Silat al-Rahim/Kinship Ties
- **Category**: social  |  **Sentiment**: positive  |  **Dialect**: MSA
- **Status**: **awaiting review**

> The duty to maintain and nurture ties with blood relatives

_A religious and social obligation; severing kinship ties carries strong moral censure across the Arab world_

### 1. The Arabic exemplars

Each should read as a natural sentence a speaker might actually say,
and should express the concept — not merely mention its name.

1. يزور أرحامه كل عيد مهما بعدت المسافات
2. خصص جزءًا من راتبه لعمته الأرملة صلةً للرحم
3. قطع سفرًا طويلًا ليعود عمه المريض
4. جمع أبناء العائلة في بيته ليتعارف الصغار على أقاربهم
5. بدأ بخالته بالهدية قبل أصدقائه
6. أصلح بين ابني عمه حفاظًا على الرحم أن تُقطع

### 2. The Arabic contrasts

These are the negative side of the extraction, and the whole method
rests on them. Each should be as close as possible to the exemplars —
same topic, same register, same kind of event — with the concept
**absent**. A contrast that changes the subject cancels nothing, and
a contrast that still carries the concept poisons the direction.

1. يقضي إجازة العيد في السفر مع أصدقائه
2. حوّل جزءًا من راتبه إلى حساب التوفير
3. اعتذر عن الزيارة لبعد المسافة وضيق الوقت
4. التقى زملاء العمل في حفل الشركة السنوي

### 3. The framing

- Is `positive` the right valence, or is this contested?
- Is `MSA` right, or is this narrower than it claims?
- Does the one-line description above match how the word is used?

### 4. English side, for reference only

Judge these only if something looks wrong; the Arabic is what matters.

<details><summary>English exemplars and contrasts</summary>

1. He visits his relatives every Eid however far the distance
2. He set aside part of his salary for his widowed aunt, keeping the tie of kinship
3. He traveled a long way to visit his sick uncle
4. He gathered the extended family at his home so the children would know their relatives
5. He gave the first gift to his aunt before his friends
6. He reconciled his two cousins to keep the kinship bond from being severed

- _He spends the Eid holiday traveling with his friends_
- _He moved part of his salary into a savings account_
- _He begged off the visit because of the distance and his schedule_
- _He met his coworkers at the annual company party_

</details>

---

## حسن الجوار — `jiwar_001`

- **English gloss**: Husn al-Jiwar/Neighborliness
- **Category**: social  |  **Sentiment**: positive  |  **Dialect**: MSA
- **Status**: **awaiting review**

> The duty of care, protection and consideration owed to neighbors

_The neighbor's right is protected in pre-Islamic custom and Islamic teaching alike; proverbs rank the neighbor above the house itself_

### 1. The Arabic exemplars

Each should read as a natural sentence a speaker might actually say,
and should express the concept — not merely mention its name.

1. تفقد جاره العجوز كل صباح قبل ذهابه إلى العمل
2. أرسلت لجيرانها من طعامها يوم طبخت
3. حرس بيت جاره الغائب كأنه بيته
4. خفض صوت المذياع مراعاةً لراحة جيرانه
5. وقف مع جاره في عزائه قبل أهله
6. قيل: الجار قبل الدار، فسأل عن الجيران قبل أن يشتري البيت

### 2. The Arabic contrasts

These are the negative side of the extraction, and the whole method
rests on them. Each should be as close as possible to the exemplars —
same topic, same register, same kind of event — with the concept
**absent**. A contrast that changes the subject cancels nothing, and
a contrast that still carries the concept poisons the direction.

1. ركّب عازلًا صوتيًا ليعمل في هدوء
2. اشترى البيت بعد فحص سعره وموقعه فقط
3. طلب من الحارس استلام طرود الشقة المجاورة
4. قدّم شكوى للبلدية بسبب موقف السيارات

### 3. The framing

- Is `positive` the right valence, or is this contested?
- Is `MSA` right, or is this narrower than it claims?
- Does the one-line description above match how the word is used?

### 4. English side, for reference only

Judge these only if something looks wrong; the Arabic is what matters.

<details><summary>English exemplars and contrasts</summary>

1. He checks on his elderly neighbor every morning before work
2. When she cooked, she sent some of the food to her neighbors
3. He watched over his absent neighbor's house as if it were his own
4. He turned the radio down out of consideration for his neighbors' rest
5. He stood by his neighbor at the funeral before the family arrived
6. As the saying goes, the neighbor before the house: he asked about the neighbors before buying the home

- _He installed soundproofing so he could work in quiet_
- _He bought the house after checking only its price and location_
- _He asked the doorman to take in the next-door parcels_
- _He filed a complaint with the municipality about the parking_

</details>

---

## الشورى — `shura_001`

- **English gloss**: Shura/Consultation
- **Category**: social  |  **Sentiment**: positive  |  **Dialect**: MSA
- **Status**: **awaiting review**

> Collective consultation before binding decisions

_A Qur'anic principle and a living practice from family councils to state institutions named after it_

### 1. The Arabic exemplars

Each should read as a natural sentence a speaker might actually say,
and should express the concept — not merely mention its name.

1. ما قطع أمرًا في العائلة حتى يستشير كبارها
2. جمع أهل الحي للتشاور في أمر المسجد قبل أي قرار
3. عرض الأمر على مجلس الشورى وأخذ برأي الأغلبية
4. استشار أهل الخبرة قبل أن يوقع الاتفاق
5. قالت له أمه: شاور أختك فالأمر يخصها قبل غيرها
6. أدار الاجتماع بالشورى فسمع من الجميع قبل أن يرجح

### 2. The Arabic contrasts

These are the negative side of the extraction, and the whole method
rests on them. Each should be as close as possible to the exemplars —
same topic, same register, same kind of event — with the concept
**absent**. A contrast that changes the subject cancels nothing, and
a contrast that still carries the concept poisons the direction.

1. أصدر قراره منفردًا وأبلغهم به في اليوم التالي
2. وقّع الاتفاق ثم أخبر شركاءه بما تم
3. حسم الأمر بقرعة سريعة لضيق الوقت
4. فوّض مستشاره ليقرر نيابة عنه

### 3. The framing

- Is `positive` the right valence, or is this contested?
- Is `MSA` right, or is this narrower than it claims?
- Does the one-line description above match how the word is used?

### 4. English side, for reference only

Judge these only if something looks wrong; the Arabic is what matters.

<details><summary>English exemplars and contrasts</summary>

1. He never settled a family matter without consulting its elders
2. He gathered the neighborhood to consult about the mosque before any decision
3. He put the matter to the consultative council and followed the majority view
4. He consulted those with expertise before signing the agreement
5. His mother told him: consult your sister, the matter concerns her first
6. He ran the meeting by consultation, hearing everyone before weighing in

- _He made the decision alone and informed them the next day_
- _He signed the agreement, then told his partners what had been done_
- _He settled it with a quick coin toss for lack of time_
- _He delegated the decision entirely to his adviser_

</details>

---

## المجلس — `majlis_001`

- **English gloss**: Majlis/Communal Gathering
- **Category**: cultural  |  **Sentiment**: positive  |  **Dialect**: MSA + Gulf
- **Status**: **awaiting review**

> The gathering space and institution where community bonds and decisions are made

_The majlis is at once a room, a ritual and an institution — listed by UNESCO as intangible cultural heritage of the Gulf and beyond_

### 1. The Arabic exemplars

Each should read as a natural sentence a speaker might actually say,
and should express the concept — not merely mention its name.

1. فتح مجلسه كل مساء لمن أراد حاجة أو حديثًا
2. دار فنجان القهوة في المجلس من اليمين كما جرت العادة
3. جلس الصغار في طرف المجلس أدبًا مع كبارهم
4. حُلّت خصومة العائلتين في مجلس الشيخ قبل أن تصل المحكمة
5. تبادلوا أخبار الحي وقصائد الشعر في مجلس الجمعة
6. قام الحاضرون لقادم المجلس حتى جلس في مكانه

### 2. The Arabic contrasts

These are the negative side of the extraction, and the whole method
rests on them. Each should be as close as possible to the exemplars —
same topic, same register, same kind of event — with the concept
**absent**. A contrast that changes the subject cancels nothing, and
a contrast that still carries the concept poisons the direction.

1. أغلق مكتبه في الخامسة وغادر إلى بيته
2. عقدوا الاجتماع عبر الفيديو واكتفوا بجدول الأعمال
3. رفعت الدعوى مباشرة إلى المحكمة المختصة
4. قرأ أخبار الحي في مجموعة الواتساب

### 3. The framing

- Is `positive` the right valence, or is this contested?
- Is `MSA + Gulf` right, or is this narrower than it claims?
- Does the one-line description above match how the word is used?

### 4. English side, for reference only

Judge these only if something looks wrong; the Arabic is what matters.

<details><summary>English exemplars and contrasts</summary>

1. He opened his majlis every evening to anyone with a need or a story
2. The coffee cup went around the majlis from the right, as custom dictates
3. The young sat at the far end of the majlis out of respect for their elders
4. The feud between the two families was settled in the sheikh's majlis before it reached court
5. They traded neighborhood news and poetry at the Friday majlis
6. Those present rose for the newcomer until he had taken his seat

- _He closed his office at five and went home_
- _They held the meeting over video and stuck to the agenda_
- _The lawsuit was filed directly with the competent court_
- _He read the neighborhood news in the WhatsApp group_

</details>

---

## الفزعة — `fazaa_001`

- **English gloss**: Faz'ah/Rallying to Aid
- **Category**: social  |  **Sentiment**: positive  |  **Dialect**: Gulf
- **Status**: **awaiting review**

> Dropping everything to rally to the aid of kin, neighbor or stranger in need

_Strongest in Bedouin and Gulf usage; distinct from ordinary charity by its immediacy and collectivity — the call itself obliges response_

### 1. The Arabic exemplars

Each should read as a natural sentence a speaker might actually say,
and should express the concept — not merely mention its name.

1. فزع له أهل الحي كلهم حين احترق بيته
2. ما إن سمعوا استغاثته حتى فزعوا إليه من كل جهة
3. فزع لأخيه في الدعوى وقال: لا يُترك واحدنا وحده
4. جمعوا الفزعة لتسديد دية أنقذت شابًا من السجن
5. ترك عشاءه وفزع لجاره حين تعطلت سيارته في البر
6. قامت القبيلة فزعةً واحدة لنصرة المظلوم

### 2. The Arabic contrasts

These are the negative side of the extraction, and the whole method
rests on them. Each should be as close as possible to the exemplars —
same topic, same register, same kind of event — with the concept
**absent**. A contrast that changes the subject cancels nothing, and
a contrast that still carries the concept poisons the direction.

1. اتصل له بشركة الطوارئ وأعطاه رقمها
2. تبرع بمبلغ شهري ثابت لجمعية خيرية
3. نصحه بتوكيل محامٍ جيد ثم انصرف
4. انتظر وصول ونش الطريق ليسحب السيارة

### 3. The framing

- Is `positive` the right valence, or is this contested?
- Is `Gulf` right, or is this narrower than it claims?
- Does the one-line description above match how the word is used?

### 4. English side, for reference only

Judge these only if something looks wrong; the Arabic is what matters.

<details><summary>English exemplars and contrasts</summary>

1. The whole neighborhood rallied to him when his house burned
2. The moment they heard his call for help they converged from every side
3. He stood with his brother in the lawsuit, saying none of us is left alone
4. They pooled the faz'ah to pay the blood money that saved a young man from prison
5. He left his dinner and went out to his neighbor whose car had broken down in the desert
6. The tribe rose as one to back the wronged man

- _He called the emergency service for him and passed on the number_
- _He set up a fixed monthly donation to a charity_
- _He advised him to hire a good lawyer, then left_
- _He waited for the tow truck to come pull the car_

</details>

---

## الحياء — `hayaa_001`

- **English gloss**: Haya'/Modesty and Propriety
- **Category**: ethical  |  **Sentiment**: mixed  |  **Dialect**: MSA
- **Status**: **awaiting review**

> A restraining sense of modesty and shame that guards conduct

_Prized as a virtue for all, though its expectations weigh unevenly by gender in practice — which keeps the label mixed rather than simply positive_

### 1. The Arabic exemplars

Each should read as a natural sentence a speaker might actually say,
and should express the concept — not merely mention its name.

1. منعه حياؤه أن يسأل الناس شيئًا مع حاجته
2. غضّت الطرف حياءً حين أُثني عليها أمام الحضور
3. استحيا أن يرفع صوته في حضرة والده
4. قال النبي إن الحياء لا يأتي إلا بخير
5. استحيت أن تأكل قبل أن يبدأ كبير المجلس
6. منعه الحياء من ذكر إنجازه فذكره غيره عنه

### 2. The Arabic contrasts

These are the negative side of the extraction, and the whole method
rests on them. Each should be as close as possible to the exemplars —
same topic, same register, same kind of event — with the concept
**absent**. A contrast that changes the subject cancels nothing, and
a contrast that still carries the concept poisons the direction.

1. عرض إنجازاته بالتفصيل في مقابلة العمل
2. طلب زيادة راتبه مباشرة في الاجتماع
3. ناقش والده بصوت مرتفع أمام الضيوف
4. بدأ بالأكل فور جلوسه إلى المائدة

### 3. The framing

- Is `mixed` the right valence, or is this contested?
- Is `MSA` right, or is this narrower than it claims?
- Does the one-line description above match how the word is used?

### 4. English side, for reference only

Judge these only if something looks wrong; the Arabic is what matters.

<details><summary>English exemplars and contrasts</summary>

1. His sense of haya' kept him from asking people for anything despite his need
2. She lowered her gaze in modesty when praised before the gathering
3. He was too respectful to raise his voice in his father's presence
4. The Prophet said that haya' brings nothing but good
5. She held back from eating before the eldest of the gathering began
6. Modesty kept him from mentioning his achievement, so others mentioned it for him

- _He laid out his achievements in detail at the job interview_
- _He asked for a raise directly in the meeting_
- _He argued loudly with his father in front of the guests_
- _He started eating the moment he sat at the table_

</details>

---
