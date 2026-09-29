# CI/CD и первый релиз — 30 сентября 2026

Публикация по умолчанию выключена. CI выполняет тесты и сборку; PyPI upload
и создание GitHub Release включаются repository variable `PYPI_PUBLISH_ENABLED=true`.
Оба каталога TTS относятся к одному GitHub-репозиторию `Atm4x/tts-with-rvc`:
их нужно публиковать в разные ветки, сохраняя разные имена пакетов.

## История за последние 7 дней

После `git fetch origin --prune` новые коммиты подтверждены на удалённых ветках:

| Каталог | Исходная ветка | Последний коммит | Дата, Москва | Новая локальная ветка |
|---|---|---|---|---|
| FSpipeline2 | multigpu | d6cfb23 — Fix for multigpu | 29.09.2026 23:53 | releases |
| tts-with-rvc-onnx-multigpu | multigpu-onnx | e831375 — MultiGPU | 27.09.2026 22:40 | releases-onnx |
| tts-with-rvc-multigpu | multigpu | 753555c — Multigpu test | 27.09.2026 22:40 | releases |

## Пакеты и изменения

| PyPI-проект | Опубликованная версия | Подготовленная версия | GitHub Environment |
|---|---|---|---|
| fish-speech-lib | 0.1.0.1 | 0.1.0.2 | pypi-fish |
| tts-with-rvc | 0.1.9.2 | 0.1.9.3 | pypi-rvc |
| tts-with-rvc-onnx | 0.1.9.4 | 0.1.9.5 | pypi-onnx |

По PyPI JSON API и содержимому опубликованных wheel это три чистых Python-пакета:
wheel `py3-none-any` и исходный архив `sdist`. В них не встроены CUDA,
ONNX Runtime, fairseq или веса моделей: это зависимости / внешние ресурсы.
В опубликованных wheel было 40 / 30 / 11 Python-файлов у Fish / RVC / ONNX;
в новых — 44 / 33 / 14. Оба TTS-пакета используют импорт `tts_with_rvc`,
поэтому устанавливать их вместе в одно окружение не следует.

У TTS заменён `setup.py` на `pyproject.toml`. Списки зависимостей сохранены
без изменений относительно текущих исходников; ONNX extras `cuda` и `dml`
также сохранены. Оба TTS объявляют Python `>=3.10,<3.13`.
Fish сохраняет `>=3.10`; сборочный backend обновлён до setuptools с поддержкой
SPDX, в архивы входят Apache LICENSE и NOTICE.

## Что делают workflows

- `.github/workflows/ci.yml`: тесты Python 3.10/3.11/3.12 на Ubuntu и Windows;
  затем изолированная сборка sdist и wheel из sdist на Ubuntu / Python 3.12.
- `.github/workflows/release.yml`: push в релизную ветку запускает тот же CI,
  проверяет площадку/ветку и новую версию, публикует готовые архивы в PyPI,
  затем создаёт GitHub Release с этими же файлами.
- Имена GitHub-тегов разделены: `fish-speech-lib/v...`, `tts-with-rvc/v...`,
  `tts-with-rvc-onnx/v...`.
- Проверяется совпадение версий в TOML и `__init__.py`, отсутствие уже
  опубликованного номера и увеличение относительно последнего стабильного
  релиза. Проверяются все Python-файлы и их содержимое, лицензии, метаданные
  wheel/sdist и установка wheel в отдельный venv без зависимостей.
- Публикация разрешена только из `releases` для Fish/RVC и `releases-onnx`
  для ONNX; случайно отправленный ONNX в `releases` остановится до загрузки.
- Actions закреплены на конкретных SHA. Публикация получает OIDC-разрешение,
  создание GitHub Release — отдельное `contents: write`.

Обычные push запускают CI; релизные push обрабатывает release workflow.
`workflow_dispatch` у CI позволяет вручную только проверить сборку.
`workflow_dispatch` у release публикует только при выборе релизной ветки.

## Авторизация и включение публикации

Push без секретов безопасен для публикации: publish job будет пропущен.
Когда авторизация готова, в **Settings → Secrets and variables → Actions →
Variables** создай repository variable `PYPI_PUBLISH_ENABLED` со значением
`true`. После этого новый push в релизную ветку запустит публикацию.
Для TTS одна repository variable действует на обе релизные ветки.

Создай GitHub Environments:

| GitHub-репозиторий | Environment | Разрешённая deployment-ветка |
|---|---|---|
| Atm4x/Fish-speech-pipeline | pypi-fish | releases |
| Atm4x/tts-with-rvc | pypi-rvc | releases |
| Atm4x/tts-with-rvc | pypi-onnx | releases-onnx |

Путь: **Settings → Environments → New environment**.
При необходимости добавь reviewer для ручного одобрения публикации.

### Вариант A: имеющиеся API-токены

В каждом Environment добавь **Environment secret** с одинаковым именем
`PYPI_API_TOKEN`, но значением соответствующего ключа.
Если используется один account-wide токен с правами на все проекты, можно
задать `PYPI_API_TOKEN` как repository secret в обоих GitHub-репозиториях.
Если используются разные project-scoped токены, задай их в соответствующих
Environments: `pypi-rvc` и `pypi-onnx`.

Из открытых полей присланных токенов удалось определить:

| Сообщение | Площадка / ограничение | Куда подходит |
|---|---|---|
| 30.03.2025 01:08 | TestPyPI, без ограничения на один проект | Для TestPyPI; текущие workflows публикуют в основной PyPI |
| 30.03.2025 02:59, первый токен | PyPI, без project restriction | Можно использовать для Fish, если аккаунт имеет права на fish-speech-lib |
| 30.03.2025 02:59, второй токен | PyPI, без project restriction | Аналогично; это отдельный токен того же аккаунта |
| 24.04.2025 17:51, ONNX PYPI | PyPI, только tts-with-rvc-onnx | Secret PYPI_API_TOKEN в pypi-onnx |
| 24.04.2025 17:56, tts-with-rvc | PyPI, только tts-with-rvc | Secret PYPI_API_TOKEN в pypi-rvc |

Recovery codes — резервные коды входа с 2FA; в GitHub Secrets для публикации
их вставлять не нужно. Сами ключи и recovery codes в файлах не сохранены.
Я локально прочитал только открытые заголовки токенов, не отправляя ключи
на сервер. Это определяет площадку и ограничения, но не подтверждает
действительность токена или актуальные права аккаунта.

Документированного read-only endpoint для проверки обычного PyPI API-токена
не нашёл. Публичный JSON API работает без авторизации и не проверяет ключ.
Если публикация вернёт 403, проверь права аккаунта и токены в Account settings.
Для нового ключа: **PyPI → Account settings → API tokens → Add API token**,
Scope — соответствующий проект; замени Environment secret `PYPI_API_TOKEN`.

### Вариант B: Trusted Publishing без API-токенов

Не задавай `PYPI_API_TOKEN` в Environment или на уровне репозитория.
В PyPI открой **Manage project → Publishing → Add a new publisher → GitHub**:

| Проект | Owner | Repository name | Workflow filename | Environment name |
|---|---|---|---|---|
| fish-speech-lib | Atm4x | Fish-speech-pipeline | release.yml | pypi-fish |
| tts-with-rvc | Atm4x | tts-with-rvc | release.yml | pypi-rvc |
| tts-with-rvc-onnx | Atm4x | tts-with-rvc | release.yml | pypi-onnx |

Если секрет задан, action использует токен. Если секрет пуст, используется
OIDC Trusted Publishing с attestations. Регистрируется именно `release.yml`,
который публикует, а не переиспользуемый `ci.yml`.

Официальные инструкции:
[Trusted Publisher](https://docs.pypi.org/trusted-publishers/adding-a-publisher/),
[API tokens](https://pypi.org/help/#apitoken),
[PyPI publish action](https://github.com/pypa/gh-action-pypi-publish).

## Первый коммит и push

Эти команды отправляют подготовленные изменения в релизные ветки.
Реальная публикация запустится только при `PYPI_PUBLISH_ENABLED=true`.
Локальные ветки уже созданы от свежих multigpu-веток.

```powershell
Set-Location 'D:\Projects\FSpipeline2'
git add -- .github pyproject.toml fish_speech_lib/__init__.py fish_speech_lib/fish_speech/diagnostics.py fish_speech_lib/fish_speech/models/text2semantic/compile_runtime.py tests/test_device_pipeline.py tests/test_public_release.py tests/test_diagnostics.py
git commit -m "Add package CI/CD and prepare fish-speech-lib 0.1.0.2"
git push -u origin releases

Set-Location 'D:\Projects\tts-with-rvc-onnx-multigpu'
git add -- .github pyproject.toml setup.py tts_with_rvc/__init__.py tests/test_architecture.py tests/test_public_api.py
git commit -m "Add package CI/CD and prepare tts-with-rvc-onnx 0.1.9.5"
git push -u origin releases-onnx

Set-Location 'D:\Projects\tts-with-rvc-multigpu'
git add -- .github pyproject.toml setup.py tts_with_rvc/__init__.py
git commit -m "Add package CI/CD and prepare tts-with-rvc 0.1.9.3"
git push -u origin releases
```

Для проверки GitHub CI до публикации можно сначала отправить каждую текущую
ветку в отдельную ветку `ci-package-check` / `ci-package-check-onnx`:

```powershell
git -C 'D:\Projects\FSpipeline2' push origin HEAD:refs/heads/ci-package-check
git -C 'D:\Projects\tts-with-rvc-multigpu' push origin HEAD:refs/heads/ci-package-check
git -C 'D:\Projects\tts-with-rvc-onnx-multigpu' push origin HEAD:refs/heads/ci-package-check-onnx
```

Эти команды требуют уже сделанных локальных коммитов; незакоммиченные файлы
push не отправляет. Для такого порядка сначала выполни только `git add` и
`git commit` из первого блока, затем проверочные push и дождись зелёного CI,
после этого — релизные push.

`gh auth status` обнаружил недействительную локальную авторизацию GitHub CLI.
При использовании CLI обнови её через `gh auth login -h github.com`.
Это отдельно от git credential manager и отдельно от встроенного GITHUB_TOKEN
на GitHub Actions: в workflow gh получает токен автоматически.

## Следующие релизы

Версия выбирается локально и попадает в обычный коммит. CI её не увеличивает
скрыто и не создаёт служебных коммитов в твоих ветках.

```powershell
python -m pip install 'packaging>=24,<27' 'tomli>=2,<3'
python .github/scripts/package.py version --bump
```

`--bump` увеличивает последний числовой компонент: `0.1.9.3 → 0.1.9.4`.
Для другого номера:

```powershell
python .github/scripts/package.py version --set 0.2.0
```

Команда синхронно обновляет TOML и `__version__`; затем включи оба файла
в коммит. При переносе новых multigpu-изменений в релизную ветку сначала
проверь конфликты упаковки: multigpu пока ещё содержит старые setup.py/версии.
Не сливай ONNX- и обычную RVC-релизные ветки друг в друга.

Если публикация прошла, а создание GitHub Release упало — перезапускай
только failed jobs. Полный повтор остановится на уже опубликованной версии.
Если загрузился только один из двух PyPI-файлов, не пытайся перезаписать его:
проверь страницу релиза и восстанови второй файл либо подготовь новый номер.

## Результаты локальной проверки

- Windows, Python 3.12.9, CPU PyTorch 2.8.0.
- Fish: 34 теста; ONNX: 17; RVC: 18. Все прошли.
- Дополнительно 8 тестов защиты релиза в каждом checkout; все прошли.
- Все три sdist и wheel собраны через изолированный `python -m build`.
- Все шесть архивов прошли `twine check --strict`.
- Проверены модули, содержимое исходников, имена/версии, Python requirement,
  LICENSE/NOTICE и установка wheel с `--no-deps` в отдельный venv.
- Все шесть YAML-workflows проверены `actionlint`.

Полная установка тяжёлых runtime-зависимостей, генерация речи с весами моделей
и реальная работа нескольких GPU здесь не проверялись. Матрица Ubuntu/Windows
и Python 3.10/3.11/3.12 выполнится на GitHub после push; локально выполнялся
Windows / Python 3.12. Публикация и серверная проверка ключей не выполнялись.

Fish по твоему уточнению сохраняет компактный
`%TEMP%\fish_speech_devices_<PID>.json`: последнюю конфигурацию компилятора
для каждого устройства. JSON атомарно заменяется при настройке компилятора,
не записывается на каждом токене и не дублируется в консоль. Исходные обычные
сообщения Fish о загрузке/генерации оставлены.

PyPI metadata:
[fish-speech-lib](https://pypi.org/pypi/fish-speech-lib/json),
[tts-with-rvc](https://pypi.org/pypi/tts-with-rvc/json),
[tts-with-rvc-onnx](https://pypi.org/pypi/tts-with-rvc-onnx/json).
