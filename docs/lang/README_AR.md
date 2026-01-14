<!---
Copyright 2026 EGen Team. All rights reserved.

Licensed under the MIT License.
-->

<div align="center">
    <img src="../../docs/assets/banner.png" alt="THL Banner" width="100%"/>
</div>
<br>

<p align="center">
    <img src="https://img.shields.io/badge/python-3.8+-blue.svg" alt="Python Version">
    <img src="https://img.shields.io/badge/license-MIT-green.svg" alt="License">
    <img src="https://img.shields.io/badge/vram-4GB-orange.svg" alt="VRAM Optimized">
    <a href="https://github.com/EGen-V/Transformer-Hierarchical-Layers/actions"><img src="https://github.com/EGen-V/Transformer-Hierarchical-Layers/workflows/Tests/badge.svg" alt="Tests"></a>
</p>

<h1 align="center">🤗 THL: Transformer Hierarchical Layers</h1>

<p align="center">
    <a href="README_AR.md">العربية</a> |
    <a href="../../README.md">English</a> |
    <a href="README_ES.md">Español</a> |
    <a href="README_FR.md">Français</a> |
    <a href="README_zh-hans.md">简体中文</a>
</p>

<h3 align="center">
    أحدث النماذج المتكررة الهرمية للأجهزة منخفضة الموارد
</h3>

<p align="center">
    <div dir="rtl">
    THL هو رسم بياني حسابي متكرر هرمي، غير معتمد على Transformer، مصمم لتشغيل النماذج اللغوية الكبيرة على <b>4GB VRAM</b> والأجهزة المحمولة.
    </div>
</p>

---

<div dir="rtl">

تحل **THL** المشكلة المحددة المتمثلة في **انفجار ذاكرة التخزين المؤقت KV** في المحولات (Transformers) باستخدام **ذاكرة مستقلة عن طول التسلسل** (O(1) ذاكرة لكل طبقة). إنها تحقق أداءً منافسًا للمحولات مع تمكين الاستنتاج على أجهزة المستهلك.

## ⚡ لماذا تستخدم THL؟

1.  **ذاكرة محدودة (O(1))**: انسَ ذاكرة التخزين المؤقت O(T) KV. تستخدم THL ذاكرة بفتحات ثابتة (`J=1024`)، مما يسمح بتوليد سياق لا نهائي دون تعطل وحدة معالجة الرسومات الخاصة بك.
2.  **تكرار هرمي**: تعالج طبقات GRU متعددة المقاييس الزمنية المعلومات بترددات مختلفة ($\tau_k$)، مما يلتقط كلاً من البنية المحلية والدلالات العالمية بكفاءة.
3.  **استنتاج منخفض VRAM**: يسمح **محرك الاستنتاج الطبقي** المدمج بتشغيل نماذج بأكثر من 7B معلمة على أقل من 4GB VRAM.
4.  **توجيه متفرق**: يضمن التوجيه Top-K متعدد الرؤوس الوصول إلى الذكريات ذات الصلة دون معالجة السجل بأكمله.

## 🛠️ التثبيت

```bash
# استنساخ المستودع
git clone https://github.com/EGen-V/Transformer-Hierarchical-Layers.git
cd Core

# تثبيت الاعتمادات
pip install -r requirements.txt
pip install .
```

## 🚀 جولة سريعة

### 1. نمذجة اللغة الأساسية

قم بإنشاء نموذج وتشغيل تمرير أمامي بسهولة:

```python
import torch
from thl.config import THLConfig
from thl.model import THLModel

# تكوين لـ 4GB VRAM
config = THLConfig(
    num_tiers=3,
    memory_slots=1024,
    dim=768
)

model = THLModel(config)
input_ids = torch.randint(0, 50257, (1, 32))
logits, state = model(input_ids)
```

### 2. التوليد بذاكرة منخفضة (دفق)

تشغيل نماذج أكبر عن طريق دفق الطبقات إلى وحدة معالجة الرسومات واحدة تلو الأخرى:

```python
from thl.inference.layered import LayeredInferenceEngine
from thl.inference.state import InferenceState

engine = LayeredInferenceEngine(model, device="cuda")
state = InferenceState.init(1, config, model.tiers, model.memory_bank)

# خطوة توليد رمز واحد
token = torch.tensor([123])
logit, state = engine.step(token, state)
```

## 🏗️ المعمارية

| المكون | الرمز | الوصف |
|-----------|---|-------------|
| **بنك الذاكرة** | $M_t$ | مصفوفة ثابتة الحجم ($J \times d$) تحتفظ بالسياق طويل المدى. |
| **الموجه المتفرق** | $r_t$ | آلية توجيه Top-K لقراءة الفتحات ذات الصلة. |
| **الطبقات الهرمية** | $s_t^{(k)}$ | كومة من الخلايا المتكررة يتم تحديثها بفاصل زمني أسي $\tau=2^k$. |
| **كاتب الجديد** | $w_t$ | آلية بوابية لكتابة المعلومات الجديدة فقط في الذاكرة. |

## 🧪 الأداء الموثق

نحن نختبر THL بدقة. قم بتشغيل المجموعة بنفسك:
```bash
./scripts/run_tests.sh
```

## 📜 الترخيص

هذا المشروع مرخص بموجب رخصة MIT.

</div>
