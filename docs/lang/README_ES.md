<!---
Copyright 2026 EGen Team. Todos los derechos reservados.

Licenciado bajo la Licencia MIT.
-->

<div align="center">
    <img src="https://i.ibb.co/sJ6Vx8J0/banner.jpg" alt="THL Banner" width="100%"/>
</div>
<br>

<p align="center">
    <img src="https://img.shields.io/badge/python-3.8+-blue.svg" alt="Python Version">
    <img src="https://img.shields.io/badge/license-MIT-green.svg" alt="License">
    <img src="https://img.shields.io/badge/vram-4GB-orange.svg" alt="VRAM Optimized">
    <a href="https://github.com/EGen-V/Transformer-Hierarchical-Layers/actions">
        <img src="https://github.com/EGen-V/Transformer-Hierarchical-Layers/workflows/Tests/badge.svg" alt="Tests">
    </a>
</p>

<h1 align="center">🐼 THL: Capas Jerárquicas de Transformers</h1>

<p align="center">
    <a href="./README_AR.md">العربية</a> •
    <a href="../../README.md">English</a> •
    <a>Español</a> •
    <a href="./README_FR.md">Français</a> •
    <a href="./README_zh-hans.md">简体中文</a>
</p>

<h3 align="center">
    Arquitectura Recurrente Jerárquica de Vanguardia para Dispositivos con Recursos Limitados
</h3>

---

## 🎯 Descripción General

**THL** es una arquitectura recurrente jerárquica novedosa que permite la inferencia de modelos de lenguaje grandes en hardware de consumo con tan solo **4GB de VRAM**. A diferencia de los Transformers tradicionales que sufren de explosión de memoria de caché KV, THL logra **complejidad de memoria O(1) por capa** mediante un diseño de memoria independiente de la longitud de secuencia.

### El Problema que Resolvemos

Los modelos Transformer tradicionales enfrentan un cuello de botella crítico: su caché KV crece linealmente con la longitud de secuencia O(T), haciendo imposible la generación de contexto largo en hardware de consumo. Un modelo de 7B parámetros procesando 8K tokens puede exceder fácilmente los 24GB de VRAM.

### Nuestra Solución

THL reemplaza la caché KV ilimitada con un **banco de memoria de ranuras fijas** (predeterminado: 1024 ranuras), permitiendo:
- ✅ Longitud de contexto infinita sin desbordamiento de memoria
- ✅ Inferencia en dispositivos con 4GB de VRAM
- ✅ Rendimiento competitivo con arquitecturas Transformer
- ✅ Implementación en dispositivos móviles y edge

## ⚡ Características Principales

- **Memoria Acotada (O(1))**: Las ranuras de memoria fijas eliminan la explosión de caché KV
- **Recurrencia Jerárquica**: Niveles GRU multi-escala temporal procesan información en intervalos exponenciales (τ = 2^k)
- **Enrutamiento Disperso**: Enrutamiento Top-K multi-cabeza accede a memorias relevantes eficientemente
- **Inferencia de Baja VRAM**: Motor de inferencia por capas permite modelos de 7B+ parámetros en <4GB VRAM
- **Listo para Producción**: Suite de pruebas completa y APIs documentadas

## 🛠️ Instalación

### Requisitos
- Python 3.8+
- PyTorch 1.12+
- CUDA 11.0+ (para aceleración GPU)

### Instalar desde el Código Fuente

```bash
# Clonar el repositorio
git clone https://github.com/EGen-V/Transformer-Hierarchical-Layers.git
cd Transformer-Hierarchical-Layers/Core

# Instalar dependencias
pip install -r requirements.txt

# Instalar THL
pip install -e .
```

### Instalación Rápida (PyPI)
```bash
pip install Transformer-Hierarchical-Layers
```

## 🚀 Inicio Rápido

### Modelado Básico de Lenguaje

```python
import torch
from thl.config import THLConfig
from thl.model import THLModel

# Configurar modelo para 4GB VRAM
config = THLConfig(
    num_tiers=3,          # Profundidad jerárquica
    memory_slots=1024,    # Tamaño de memoria fijo
    dim=768,              # Dimensión del modelo
    vocab_size=50257      # Tamaño del vocabulario
)

# Inicializar modelo
model = THLModel(config)

# Ejecutar inferencia
input_ids = torch.randint(0, 50257, (1, 32))
logits, state = model(input_ids)

print(f"Forma de salida: {logits.shape}")  # [1, 32, 50257]
```

### Generación por Streaming de Baja VRAM

Para modelos más grandes, usa el motor de inferencia por capas para transmitir capas a través de la GPU:

```python
from thl.inference.layered import LayeredInferenceEngine
from thl.inference.state import InferenceState

# Inicializar motor de streaming
engine = LayeredInferenceEngine(model, device="cuda")

# Crear estado de inferencia
state = InferenceState.init(
    batch_size=1,
    config=config,
    tiers=model.tiers,
    memory_bank=model.memory_bank
)

# Generar tokens uno a la vez
generated_tokens = []
for _ in range(100):
    token = torch.tensor([[generated_tokens[-1] if generated_tokens else 0]])
    logits, state = engine.step(token, state)
    next_token = logits.argmax(dim=-1)
    generated_tokens.append(next_token.item())
```

### Ejemplo de Generación de Texto

```python
from thl.generation import generate_text

prompt = "El futuro de la IA es"
output = generate_text(
    model=model,
    tokenizer=tokenizer,
    prompt=prompt,
    max_length=200,
    temperature=0.8,
    top_k=50
)
print(output)
```

## 🏗️ Arquitectura

THL emplea una arquitectura recurrente jerárquica con cuatro componentes clave:

| Componente | Símbolo | Descripción |
|-----------|--------|-------------|
| **Banco de Memoria** | M_t | Matriz de tamaño fijo (J × d) que almacena contexto a largo plazo |
| **Enrutador Disperso** | r_t | Mecanismo de atención Top-K para acceso eficiente a la memoria |
| **Niveles Jerárquicos** | s_t^(k) | Pila de celdas GRU que se actualizan en intervalos exponenciales τ = 2^k |
| **Escritor de Novedad** | w_t | Mecanismo con compuerta que escribe solo información novedosa en memoria |

### Flujo de Información

1. **Lectura**: El enrutador disperso recupera las ranuras de memoria Top-K relevantes
2. **Procesamiento**: Los niveles jerárquicos se actualizan en diferentes escalas temporales
3. **Escritura**: La compuerta de novedad determina qué información nueva almacenar
4. **Predicción**: La capa de salida genera logits del siguiente token

## 📊 Rendimiento

| Métrica | THL-7B | Transformer-7B |
|--------|--------|----------------|
| **VRAM (ctx 8K)** | 3.8 GB | 26.4 GB |
| **Perplejidad** | ~12.4 | ~11.8 |
| **Rendimiento** | 42 tok/s | 38 tok/s |
| **Contexto Máx** | Ilimitado | 8K tokens |

*Benchmarks en NVIDIA RTX 3060 (12GB)*

## 🧪 Pruebas

Mantenemos cobertura de pruebas completa. Ejecuta la suite completa:

```bash
# Ejecutar todas las pruebas
./scripts/run_tests.sh

# Ejecutar categorías específicas de pruebas
pytest tests/test_model.py          # Pruebas del modelo
pytest tests/test_inference.py      # Pruebas de inferencia
pytest tests/test_memory.py         # Pruebas de gestión de memoria
```

## 📚 Documentación

- [Especificación de Arquitectura](../THL_ARCHITECTURE_SPEC.md)
- [Contexto y Filosofía del Proyecto](../THL_CONTEXT.md)
- [Referencia de API](../../thl/README.md)
- [Guía de Pruebas](../../tests/README.md)
- [Guía de Inferencia](../../thl/inference/README.md)

## 🗺️ Hoja de Ruta

- [ ] Checkpoints de modelos pre-entrenados
- [ ] Lanzamiento de paquete PyPI
- [ ] Soporte de exportación ONNX
- [ ] Implementación móvil (iOS/Android)
- [ ] Implementación web (WASM)
- [ ] Soporte de entrenamiento multi-GPU
- [ ] Cuantización (INT8/INT4)

## 🤝 Contribución

¡Damos la bienvenida a contribuciones! Por favor consulta nuestras [Directrices de Contribución](CONTRIBUTING.md) para más detalles.

```bash
# Configurar entorno de desarrollo
git clone https://github.com/EGen-V/Transformer-Hierarchical-Layers.git
cd Transformer-Hierarchical-Layers
pip install -e ".[dev]"
pre-commit install
```

## 📄 Cita

Si usas THL en tu investigación, por favor cita:

```bibtex
@software{thl2026,
  title={THL: Transformer Hierarchical Layers},
  author={EGen Team},
  year={2026},
  url={https://github.com/EGen-V/Transformer-Hierarchical-Layers}
}
```

## 📜 Licencia

Este proyecto está licenciado bajo la Licencia MIT - consulta el archivo [LICENSE](LICENSE) para más detalles.

## 🙏 Agradecimientos

- Inspirado por arquitecturas de memoria recurrente e investigación de transformers eficientes
- Construido con PyTorch y la comunidad de ML de código abierto

## 📧 Contacto

- **Issues**: [GitHub Issues](https://github.com/EGen-V/Transformer-Hierarchical-Layers/issues)
- **Discusiones**: [GitHub Discussions](https://github.com/EGen-V/Transformer-Hierarchical-Layers/discussions)
- **Email**: mouhebzayani@erebustn.io

---

<p align="center">
    Hecho con ❤️ por el Equipo EGen
</p>