"""The Generator's queue: numbered entries, Add current, Remove, Clear, at most MAX_QUEUE."""

from anima_trainer.preview.generator_settings import MAX_QUEUE, GenerationEntry
from anima_trainer.ui.widgets.queue_list import QueueList


def _queue(qtbot):
    queue = QueueList()
    qtbot.addWidget(queue)
    return queue


def _texts(queue):
    return [queue.list.item(row).text() for row in range(queue.list.count())]


def test_entries_round_trip_and_are_shown_numbered(qtbot):
    queue = _queue(qtbot)
    entries = (GenerationEntry(prompt="a cat"), GenerationEntry(prompt="a dog"))

    queue.set_entries(entries)

    assert queue.entries() == entries
    assert _texts(queue)[1].startswith("2. a dog")


def test_add_takes_up_to_the_maximum(qtbot):
    queue = _queue(qtbot)
    for number in range(MAX_QUEUE):
        assert queue.add(GenerationEntry(prompt=f"p{number}"))

    assert not queue.add(GenerationEntry(prompt="one more"))
    assert len(queue.entries()) == MAX_QUEUE
    assert not queue.add_button.isEnabled()


def test_remove_takes_the_selected_entry_and_renumbers(qtbot):
    queue = _queue(qtbot)
    queue.set_entries(tuple(GenerationEntry(prompt=name) for name in ("a", "b", "c")))
    queue.list.setCurrentRow(0)

    queue.remove_button.click()

    assert [entry.prompt for entry in queue.entries()] == ["b", "c"]
    assert _texts(queue)[0].startswith("1. b")


def test_clear_empties_the_queue(qtbot):
    queue = _queue(qtbot)
    queue.set_entries((GenerationEntry(prompt="a"),))

    queue.clear_button.click()

    assert queue.entries() == ()
    assert not queue.clear_button.isEnabled()


def test_while_generating_the_queue_cannot_change(qtbot):
    queue = _queue(qtbot)
    queue.set_entries((GenerationEntry(prompt="a"),))

    queue.set_editable(False)

    assert not any(
        button.isEnabled() for button in (queue.add_button, queue.remove_button, queue.clear_button)
    )
