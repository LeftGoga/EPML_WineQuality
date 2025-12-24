# Отчет об эксперименте: WineQuality_Boosting_[n_est=100_depth=2_lr=0.100]_Wine Quality - Boosting

**Дата генерации:** 2025-12-24 02:28:16

## Основная информация

- **ID задачи:** `aeb5b1e9b2374ced9bfdee23e3bc09d1`
- **Проект:** Wine Quality
- **Статус:** completed
- **Создано:** 2025-12-22T17:53:55.681000+00:00
- **Завершено:** 2025-12-23T18:47:12.623000+00:00
- **Теги:** framework:sklearn, model_type:boosting, task:classification, wine_quality

## Параметры эксперимента

| Параметр | Значение |
|----------|----------|
| `Args/boosting_learning_rate` | `0.1` |
| `Args/boosting_max_depth` | `2` |
| `Args/boosting_n_estimators` | `100` |
| `Args/experiment_name` | `` |
| `Args/mlp_hidden_layer_sizes` | `` |
| `Args/mlp_max_iter` | `` |
| `Args/model_type` | `boosting` |
| `Args/project_name` | `Wine Quality` |
| `Args/quality_threshold` | `7` |
| `Args/random_state` | `42` |
| `Args/rf_max_depth` | `` |
| `Args/rf_n_estimators` | `` |
| `Args/test_size` | `0.2` |
| `General/boosting_learning_rate` | `0.1` |
| `General/boosting_max_depth` | `2` |
| `General/boosting_n_estimators` | `100` |
| `General/model_type` | `boosting` |
| `General/n_features` | `13` |
| `General/n_test` | `320` |
| `General/n_train` | `1279` |
| `General/quality_threshold` | `7` |
| `General/random_state` | `42` |
| `General/test_size` | `0.2` |

## Метрики


| Метрика | Значение |
|---------|----------|
| `Metrics/accuracy` | `0.8969` |
| `Metrics/f1_weighted` | `0.8801` |

## Сравнение экспериментов

### Сравнительная таблица метрик

| Эксперимент | Metrics/accuracy | Metrics/f1_weighted | count | first_value | first_value_iteration | max_value | max_value_iteration | mean_value | min_value | min_value_iteration | value |
|---|---|---|---|---|---|---|---|---|---|---|---|
| WineQuality_Boosting_[n_est=100_depth=2_lr=0.100]_ | 0.8969 | 0.8801 | 2.0000 | 0.8969 | 0.0000 | 0.8969 | 0.0000 | 0.9000 | 0.8969 | 0.0000 | 0.8969 |
| WineQuality_Randomforest_Wine Quality - Random For | 0.9375 | 0.9331 | 1.0000 | 0.9375 | 0.0000 | 0.9375 | 0.0000 | 0.9400 | 0.9375 | 0.0000 | 0.9375 |

### График сравнения метрик

![Сравнение метрик](metrics_comparison.png)

## Визуализации

>![метрики](./pics/report_rf_graph1.png)
>![метрики](./pics/report_rf_graph2.png)

>![метрики](./pics/report_boost_graph1.png)
>![метрики](./pics/report_boost_graph2.png)
