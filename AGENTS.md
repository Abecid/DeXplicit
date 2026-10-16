# DeXplicit public release requirements

For all DeXplicit work, use **DeXplicit 3D model** for the released 3D predictor
and **DeXplicit renderer** for the released video model. Use these names in
papers, project pages, tweets, GitHub, model cards, checkpoint manifests,
filenames, examples, figures, videos, and user-facing logs. Keep private
experiment and training-run labels out of public release artifacts.

The standard inference and renderer-training paths must use 3D forecasts from
the DeXplicit 3D model as renderer conditions. Recorded future geometry and RGB
may be supervision, but must not enter those conditioning inputs. Follow the user-approved release wording. Do not restore removed conditioning
labels, historical-training narration, or the deleted model-card document.
Do not relabel old experimental results as newly evaluated predictions.

Before changing repository visibility, audit historical Git and model-Hub revisions as well as current files for private experiment labels. Keep the private provenance backup outside the public release.
