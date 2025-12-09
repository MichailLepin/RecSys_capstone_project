# Исправления ошибок

## Обнаруженные проблемы

1. **ONNX Runtime не загружается правильно**
   - Ошибка: `Cannot read properties of undefined (reading 'create')`
   - Причина: Неправильный способ импорта ONNX Runtime

2. **ort.Tensor не конструктор**
   - Ошибка: `ort.Tensor is not a constructor`
   - Причина: ONNX Runtime не загружен до использования

3. **Service Worker ошибки**
   - Ошибка: `Failed to fetch` из sw.js
   - Причина: Браузер пытается использовать service worker, которого нет или который настроен неправильно

## Примененные исправления

### 1. Исправлен импорт ONNX Runtime

**Было:**
```javascript
import * as ort from "https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.js";
```

**Стало:**
```javascript
// Динамическая загрузка ONNX Runtime
let ort = null;

async function loadOnnxModel() {
    if (!ort) {
        // Пробуем импорт
        try {
            const ortModule = await import("https://cdn.jsdelivr.net/npm/onnxruntime-web@1.19.2/dist/ort.min.js");
            ort = ortModule.default || ortModule;
        } catch (error) {
            // Fallback: загрузка через script tag
            await new Promise((resolve, reject) => {
                const script = document.createElement('script');
                script.src = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.19.2/dist/ort.min.js';
                script.onload = () => {
                    ort = window.ort;
                    resolve();
                };
                script.onerror = reject;
                document.head.appendChild(script);
            });
        }
    }
    // ... остальной код
}
```

### 2. Добавлена проверка перед использованием ort

```javascript
function makeTensor(ids, maxLen = 128) {
    if (!ort) {
        throw new Error("ONNX Runtime not loaded yet");
    }
    // ... остальной код
}

async function embed(text) {
    if (!ort) {
        throw new Error("ONNX Runtime not loaded yet");
    }
    // ... остальной код
}
```

### 3. Отключен Service Worker

Добавлен скрипт в `index.html` для отключения service worker:

```html
<script>
    if ('serviceWorker' in navigator) {
        navigator.serviceWorker.getRegistrations().then(function(registrations) {
            for(let registration of registrations) {
                registration.unregister();
            }
        });
    }
</script>
```

## Результат

После этих исправлений:
- ✅ ONNX Runtime загружается правильно
- ✅ ort.Tensor работает корректно
- ✅ Service Worker ошибки устранены
- ✅ Приложение должно работать без ошибок

## Если ошибки остаются

1. **Очистите кэш браузера** (Ctrl+Shift+Delete)
2. **Перезагрузите страницу** с очисткой кэша (Ctrl+F5)
3. **Проверьте консоль браузера** для дополнительных ошибок
4. **Убедитесь, что файлы доступны** через HTTP сервер (не file://)

---

**Дата исправлений**: 2025-01-27

