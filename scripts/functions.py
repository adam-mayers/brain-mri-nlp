"""
Custom function to force wandb logging within cross validation, as cannot be called within the kfold script

Adapted from https://github.com/explosion/spacy-loggers
- includes the entirety of spacy-loggers/util.py
- majority of spacy-loggers/wandb.py to include wand_logger_v5, which otherwise did not work

"""

from typing import Dict, Any, Tuple, Callable, List, IO, Optional, Iterator
from types import ModuleType
import sys
import spacy

from spacy import Language
from spacy.util import SimpleFrozenList

LoggerT = Callable[
    [Language, IO, IO],
    Tuple[Callable[[Optional[Dict[str, Any]]], None], Callable[[], None]],
]


def walk_dict(
    node: Dict[str, Any], parent: List[str] = []
) -> Iterator[Tuple[List[str], Any]]:
    """Walk a dict and yield the path and values of the leaves."""
    for key, value in node.items():
        key_parent = [*parent, key]
        if isinstance(value, dict):
            yield from walk_dict(value, key_parent)
        else:
            yield (key_parent, value)


def dot_to_dict(values: Dict[str, Any]) -> Dict[str, dict]:
    """Convert dot notation to a dict. For example: {"token.pos": True,
    "token._.xyz": True} becomes {"token": {"pos": True, "_": {"xyz": True }}}.
    values (Dict[str, Any]): The key/value pairs to convert.
    RETURNS (Dict[str, dict]): The converted values.
    """
    result = {}
    for key, value in values.items():
        path = result
        parts = key.lower().split(".")
        for i, item in enumerate(parts):
            is_last = i == len(parts) - 1
            path = path.setdefault(item, value if is_last else {})
    return result


def dict_to_dot(obj: Dict[str, dict]) -> Dict[str, Any]:
    """Convert dot notation to a dict. For example: {"token": {"pos": True,
    "_": {"xyz": True }}} becomes {"token.pos": True, "token._.xyz": True}.
    values (Dict[str, dict]): The dict to convert.
    RETURNS (Dict[str, Any]): The key/value pairs.
    """
    return {".".join(key): value for key, value in walk_dict(obj)}


def matcher_for_regex_patterns(
    regexps: Optional[List[str]] = None,
) -> Callable[[str], bool]:
    try:
        compiled = []
        if regexps is not None:
            for regex in regexps:
                compiled.append(re.compile(regex, flags=re.MULTILINE))
    except re.error as err:
        raise ValueError(
            f"Regular expression `{regex}` couldn't be compiled for logger stats matcher"
        ) from err

    def is_match(string: str) -> bool:
        for regex in compiled:
            if regex.search(string):
                return True
        return False

    return is_match


def setup_default_console_logger(
    nlp: "Language", stdout: IO = sys.stdout, stderr: IO = sys.stderr
) -> Tuple[Callable, Callable]:
    console_logger = registry.get("loggers", "spacy.ConsoleLogger.v1")
    console = console_logger(progress_bar=True)
    console_log_step, console_finalize = console(nlp, stdout, stderr)
    return console_log_step, console_finalize



# changed this to add it to registry
@spacy.registry.loggers("my_custom_logger.v1")
def wandb_logger_v5(
    project_name: str,
    remove_config_values: List[str] = SimpleFrozenList(),
    model_log_interval: Optional[int] = None,
    log_dataset_dir: Optional[str] = None,
    entity: Optional[str] = None,
    run_name: Optional[str] = None,
    log_best_dir: Optional[str] = None,
    log_latest_dir: Optional[str] = None,
    log_custom_stats: Optional[List[str]] = None,
) -> LoggerT:
    """Creates a logger that interoperates with the Weights & Biases framework.

    Args:
        project_name (str):
            The name of the project in the Weights & Biases interface. The project will be created automatically if it doesn't exist yet.
        remove_config_values (List[str]):
            A list of values to exclude from the config before it is uploaded to W&B. Defaults to [].
        model_log_interval (Optional[int]):
            Steps to wait between logging model checkpoints to the W&B dasboard. Defaults to None.
        log_dataset_dir (Optional[str]):
            Directory containing the dataset to be logged and versioned as a W&B artifact. Defaults to None.
        entity (Optional[str]):
            An entity is a username or team name where you're sending runs. If you don't specify an entity, the run will be sent to your default entity, which is usually your username. Defaults to None.
        run_name (Optional[str]):
            The name of the run. If you don't specify a run name, the name will be created by the `wandb` library. Defaults to None.
        log_best_dir (Optional[str]):
            Directory containing the best trained model as saved by spaCy, to be logged and versioned as a W&B artifact. Defaults to None.
        log_latest_dir (Optional[str]):
            Directory containing the latest trained model as saved by spaCy, to be logged and versioned as a W&B artifact. Defaults to None.
        log_custom_stats (Optional[List[str]]):
            A list of regular expressions that will be applied to the info dictionary passed to the logger. Statistics and metrics that match these regexps will be automatically logged. Defaults to None.

    Returns:
        LoggerT: Logger instance.
    """
    wandb = _import_wandb()

    def setup_logger(
        nlp: "Language", stdout: IO = sys.stdout, stderr: IO = sys.stderr
    ) -> Tuple[Callable[[Dict[str, Any]], None], Callable[[], None]]:
        match_stat = matcher_for_regex_patterns(log_custom_stats)
        run = _setup_wandb(
            wandb,
            nlp,
            project_name,
            remove_config_values=remove_config_values,
            entity=entity,
        )
        if run_name:
            wandb.run.name = run_name

        if log_dataset_dir:
            _log_dir_artifact(
                wandb, path=log_dataset_dir, name="dataset", type="dataset"
            )

        def log_step(info: Optional[Dict[str, Any]]):
            _log_scores(wandb, info)
            _log_model_artifact(wandb, info, run, model_log_interval)
            _log_custom_stats(wandb, info, match_stat)

        def finalize() -> None:
            if log_best_dir:
                _log_dir_artifact(
                    wandb,
                    path=log_best_dir,
                    name="model_best",
                    type="model",
                )

            if log_latest_dir:
                _log_dir_artifact(
                    wandb,
                    path=log_latest_dir,
                    name="model_last",
                    type="model",
                )
            wandb.join()

        return log_step, finalize

    return setup_logger


def _import_wandb() -> ModuleType:
    try:
        import wandb

        # test that these are available
        from wandb import init, log, join  # noqa: F401

        return wandb
    except ImportError:
        raise ImportError(
            "The 'wandb' library could not be found - did you install it? "
            "Alternatively, specify the 'ConsoleLogger' in the "
            "'training.logger' config section, instead of the 'WandbLogger'."
        )


def _setup_wandb(
    wandb: ModuleType,
    nlp: "Language",
    project: str,
    entity: Optional[str] = None,
    remove_config_values: List[str] = SimpleFrozenList(),
) -> Any:
    config = nlp.config.interpolate()
    config_dot = dict_to_dot(config)
    for field in remove_config_values:
        del config_dot[field]
    config = dot_to_dict(config_dot)
    run = wandb.init(project=project, config=config, entity=entity, reinit=True)
    return run


def _log_scores(wandb: ModuleType, info: Optional[Dict[str, Any]]):
    if info is not None:
        score = info["score"]
        other_scores = info["other_scores"]
        losses = info["losses"]
        wandb.log({"score": score})
        if losses:
            wandb.log({f"loss_{k}": v for k, v in losses.items()})
        if isinstance(other_scores, dict):
            wandb.log(other_scores)


def _log_model_artifact(
    wandb: ModuleType,
    info: Optional[Dict[str, Any]],
    run: Any,
    model_log_interval: Optional[int] = None,
):
    if info is not None:
        if model_log_interval and info.get("output_path"):
            if info["step"] % model_log_interval == 0 and info["step"] != 0:
                _log_dir_artifact(
                    wandb,
                    path=info["output_path"],
                    name="pipeline_" + run.id,
                    type="checkpoint",
                    metadata=info,
                    aliases=[
                        f"epoch {info['epoch']} step {info['step']}",
                        "latest",
                        "best" if info["score"] == max(info["checkpoints"])[0] else "",
                    ],
                )


def _log_dir_artifact(
    wandb: ModuleType,
    path: str,
    name: str,
    type: str,
    metadata: Optional[Dict[str, Any]] = None,
    aliases: Optional[List[str]] = None,
):
    dataset_artifact = wandb.Artifact(name, type=type, metadata=metadata)
    dataset_artifact.add_dir(path, name=name)
    wandb.log_artifact(dataset_artifact, aliases=aliases)


def _log_custom_stats(
    wandb: ModuleType, info: Optional[Dict[str, Any]], matcher: Callable[[str], bool]
):
    if info is not None:
        for k, v in info.items():
            if matcher(k):
                wandb.log({k: v})