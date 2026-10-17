from src.visualization.curves import plot_loss_accuracy


def test_accuracy_curve_uses_counting_history_and_percent_scale(tmp_path, monkeypatch):
    from src.visualization import curves

    saved = {}
    real_save = curves._save

    def capture(fig, path):
        saved[path.name] = fig
        real_save(fig, path)

    monkeypatch.setattr(curves, '_save', capture)
    plot_loss_accuracy([
        {'epoch': 0, 'train_loss': 2.0},
        {'epoch': 1, 'train_loss': 1.0, 'val_acc_exact': 0.5,
         'val_acc_within_1': 0.75, 'val_acc_within_3': 1.0},
    ], tmp_path)
    lines = saved['accuracy_curves.png'].axes[0].lines
    assert lines[0].get_ydata()[1] == 50.0
    assert lines[1].get_ydata()[1] == 75.0
    assert lines[2].get_ydata()[1] == 100.0
    assert (tmp_path / 'loss_curves.png').exists()


def test_legacy_history_does_not_invent_accuracy(tmp_path):
    plot_loss_accuracy([{'epoch': 0, 'train_loss': 1.0, 'val_map50': 0.9}], tmp_path)
    assert (tmp_path / 'loss_curves.png').exists()
    assert not (tmp_path / 'accuracy_curves.png').exists()
