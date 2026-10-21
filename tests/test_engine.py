from gesture_remote.actions import ButtonDown, ButtonUp, Command, MoveCursor, Scroll, ZoomStep
from gesture_remote.config import TRACKPAD_MAPPINGS, Settings
from gesture_remote.engine import GestureEngine, State

from synthetic import CLICK, FIST, RIGHT_CLICK, SPREAD, THREE, TWO, Session, hand


def session(**overrides):
    settings = Settings(calibrated=True, **overrides)
    return Session(GestureEngine(settings, (1440, 900)))


def of(actions, kind):
    return [a for a in actions if isinstance(a, kind)]


# --- Activation ---

def test_nothing_happens_until_an_open_hand_is_held():
    s = session()
    s.glide(0.6, (0.3, 0.5), (0.6, 0.5))
    assert s.status.state == State.ARMING
    assert of(s.actions, MoveCursor) == []


def test_holding_an_open_hand_takes_control_at_the_real_cursor():
    s = session()
    s.arm()
    assert s.status.state == State.POINTER
    s.glide(0.5, (0.5, 0.5), (0.6, 0.5))
    moves = of(s.actions, MoveCursor)
    assert moves and moves[0].x > 700  # Continues from the real cursor, moving right


def test_arming_resets_if_the_hand_closes():
    s = session()
    s.hold(0.7)
    s.hold(0.2, **FIST)
    s.hold(0.7)
    assert s.status.state == State.ARMING


# --- Clicks ---

def test_quick_index_tap_is_one_left_click_at_the_spot_before_the_bend():
    s = session()
    s.arm()
    s.hold(0.3)
    before = s.engine.cursor
    s.hold(0.1, y=0.53, **CLICK)  # The palm drifts down a little while the finger bends
    s.hold(0.4, y=0.53)
    downs, ups = of(s.actions, ButtonDown), of(s.actions, ButtonUp)
    assert [(d.button, d.clicks) for d in downs] == [("left", 1)]
    assert len(ups) == 1
    assert abs(downs[0].x - before[0]) < 1 and abs(downs[0].y - before[1]) < 1


def test_two_quick_taps_make_a_double_click():
    s = session()
    s.arm()
    s.hold(0.2)
    s.hold(0.1, **CLICK)
    s.hold(0.15)
    s.hold(0.1, **CLICK)
    s.hold(0.3)
    assert [d.clicks for d in of(s.actions, ButtonDown)] == [1, 2]


def test_middle_bend_is_a_right_click():
    s = session()
    s.arm()
    s.hold(0.1, **RIGHT_CLICK)
    s.hold(0.3)
    assert [d.button for d in of(s.actions, ButtonDown)] == ["right"]


def test_holding_the_bend_and_moving_drags_then_releases():
    s = session()
    s.arm()
    s.hold(0.3, **CLICK)
    assert s.status.state == State.CLICKING
    s.glide(0.6, (0.5, 0.5), (0.7, 0.5), **CLICK)
    assert s.status.state == State.DRAGGING
    s.hold(0.2, x=0.7)
    assert len(of(s.actions, ButtonDown)) == 1 and len(of(s.actions, ButtonUp)) == 1
    assert s.engine.metrics.counts["drags"] == 1


def test_making_a_fist_never_clicks_and_parks_the_cursor():
    s = session()
    s.arm()
    s.frame([hand(0.5, 0.5, bends=(60, 20, 10, 10))])  # Index first...
    s.hold(0.5, **FIST)  # ...then the whole hand
    assert of(s.actions, ButtonDown) == []
    assert s.status.state == State.PARKED
    parked_at = s.engine.cursor
    s.glide(0.5, (0.5, 0.5), (0.2, 0.5), **FIST)  # Reposition the hand while parked
    assert s.engine.cursor == parked_at
    s.hold(0.3, x=0.2)
    assert s.status.state == State.POINTER


# --- Tracking loss, pause, cancel ---

def test_short_dropout_mid_drag_keeps_the_button_held():
    s = session()
    s.arm()
    s.hold(0.4, **CLICK)
    s.nothing(0.15)
    assert s.status.state == State.LOST
    assert of(s.actions, ButtonUp) == []
    s.hold(0.1, **CLICK)
    assert s.status.state in (State.CLICKING, State.DRAGGING)


def test_long_loss_releases_buttons_then_switches_control_off():
    s = session()
    s.arm()
    s.hold(0.4, **CLICK)
    s.nothing(0.5)
    assert len(of(s.actions, ButtonUp)) == 1
    s.nothing(3.0)
    assert s.status.state == State.IDLE
    s.hold(0.3)
    assert s.status.state == State.ARMING  # Has to be re-armed deliberately


def test_pause_releases_held_buttons_and_ignores_the_hand():
    s = session()
    s.arm()
    s.hold(0.4, **CLICK)
    released = s.engine.pause()
    assert of(released, ButtonUp)
    s.glide(1.0, (0.5, 0.5), (0.8, 0.5))
    assert s.status.state == State.PAUSED
    s.engine.resume()
    s.hold(0.2)
    assert s.status.state == State.ARMING


def test_cancel_releases_and_disarms():
    s = session()
    s.arm()
    s.hold(0.4, **CLICK)
    assert of(s.engine.cancel(), ButtonUp)
    assert s.engine.state == State.IDLE


def test_single_frame_glitch_does_not_move_the_cursor():
    s = session()
    s.arm()
    s.hold(0.2)
    before = len(of(s.actions, MoveCursor))
    s.frame([hand(0.95, 0.1)])  # Detection jumps across the image for one frame
    s.hold(0.2)
    moves = of(s.actions, MoveCursor)[before:]
    assert all(abs(m.x - 700) < 5 for m in moves)


def test_hand_on_a_face_is_ignored():
    s = session()
    face = (0.4, 0.4, 0.3, 0.4)
    for _ in range(60):
        s.frame([hand(0.5, 0.5)], faces=[face])
    assert s.status.state == State.IDLE
    assert s.engine.metrics.counts["face_rejections"] == 60


def test_low_confidence_hands_are_ignored():
    s = session()
    s.hold(2.0, score=0.2)
    assert s.status.state == State.IDLE


# --- Scrolling and opt-in gestures ---

def test_two_fingers_scroll_with_natural_direction():
    s = session()
    s.arm()
    s.hold(0.3, **TWO)
    assert s.status.state == State.SCROLLING
    s.glide(0.6, (0.5, 0.5), (0.5, 0.6), **TWO)  # Hand moves down
    scrolls = of(s.actions, Scroll)
    assert scrolls and sum(sc.dy for sc in scrolls) > 0  # Content follows the hand
    assert of(s.actions, Command) == []


def test_flick_back_only_fires_when_mapped():
    off = session()
    off.arm()
    off.hold(0.3, **TWO)
    off.glide(0.15, (0.4, 0.5), (0.7, 0.5), **TWO)
    assert of(off.actions, Command) == []

    on = session(mappings={"two_finger_flick_right": "back"})
    on.arm()
    on.hold(0.3, **TWO)
    on.glide(0.15, (0.4, 0.5), (0.7, 0.5), **TWO)
    assert [c.name for c in of(on.actions, Command)] == ["back"]


def test_spread_hold_runs_mission_control_only_when_mapped():
    off = session()
    off.arm()
    off.hold(1.0, **SPREAD)
    assert of(off.actions, Command) == []
    assert off.status.state == State.POINTER

    on = session(mappings=dict(TRACKPAD_MAPPINGS))
    on.arm()
    on.hold(1.0, **SPREAD)
    assert [c.name for c in of(on.actions, Command)] == ["mission_control"]


def test_spread_flick_switches_desktop_when_mapped():
    s = session(mappings=dict(TRACKPAD_MAPPINGS))
    s.arm()
    s.hold(0.25, **SPREAD)
    s.glide(0.15, (0.6, 0.5), (0.35, 0.5), **SPREAD)
    assert [c.name for c in of(s.actions, Command)] == ["next_desktop"]


def test_three_finger_drag_is_opt_in():
    off = session()
    off.arm()
    off.hold(0.5, **THREE)
    assert of(off.actions, ButtonDown) == []

    on = session(three_finger_drag=True)
    on.arm()
    on.hold(0.3, **THREE)
    on.glide(0.4, (0.5, 0.5), (0.6, 0.5), **THREE)
    assert on.status.state == State.DRAGGING
    on.hold(0.3)
    assert len(of(on.actions, ButtonDown)) == 1 and len(of(on.actions, ButtonUp)) == 1


# --- Two-hand zoom ---

def test_two_open_hands_moving_apart_zoom_in_and_together_zoom_out():
    s = session()
    s.arm()
    s.two_hands(0.3, 0.3, 0.3)
    assert s.status.state == State.ZOOMING
    s.two_hands(0.6, 0.3, 0.6)
    steps_in = sum(z.direction for z in of(s.actions, ZoomStep))
    assert steps_in > 0
    s.two_hands(0.6, 0.6, 0.3)
    assert sum(z.direction for z in of(s.actions, ZoomStep)) == 0  # Back to where it started


def test_zoom_needs_deliberate_activation():
    s = session()
    s.two_hands(1.0, 0.3, 0.6)
    assert of(s.actions, ZoomStep) == []


def test_small_hand_wobble_inside_the_dead_zone_does_not_zoom():
    s = session()
    s.arm()
    s.two_hands(0.3, 0.3, 0.3)
    for _ in range(5):
        s.two_hands(0.1, 0.3, 0.31)
        s.two_hands(0.1, 0.31, 0.3)
    assert of(s.actions, ZoomStep) == []


def test_zoom_is_bounded():
    s = session(zoom_max_steps=3)
    s.arm()
    s.two_hands(0.3, 0.2, 0.2)
    s.two_hands(1.0, 0.2, 0.9)
    assert sum(z.direction for z in of(s.actions, ZoomStep)) == 3


def test_closing_both_hands_clutches_so_they_can_be_recentred():
    s = session()
    s.arm()
    s.two_hands(0.3, 0.3, 0.3)
    s.two_hands(0.5, 0.3, 0.55)
    level = sum(z.direction for z in of(s.actions, ZoomStep))
    s.two_hands(0.5, 0.55, 0.3, FIST, FIST)  # Bring the hands back together while closed
    assert sum(z.direction for z in of(s.actions, ZoomStep)) == level
    s.two_hands(0.3, 0.3, 0.3)
    s.two_hands(0.5, 0.3, 0.55)  # Opening again continues zooming in from the current level
    assert sum(z.direction for z in of(s.actions, ZoomStep)) > level


def test_zoom_ends_when_a_hand_drops_and_pointer_continues():
    s = session()
    s.arm()
    s.two_hands(0.3, 0.3, 0.3)
    s.hold(0.4)
    assert s.status.state == State.POINTER


def test_ctrl_scroll_zoom_is_continuous():
    s = session(zoom_output="ctrl_scroll")
    s.arm()
    s.two_hands(0.3, 0.3, 0.3)
    s.two_hands(0.6, 0.3, 0.6)
    scrolls = of(s.actions, Scroll)
    assert scrolls and all(sc.zoom for sc in scrolls) and sum(sc.dy for sc in scrolls) > 0
    assert of(s.actions, ZoomStep) == []


# --- Determinism ---

def test_replaying_the_same_frames_gives_the_same_actions():
    def run():
        s = session(mappings=dict(TRACKPAD_MAPPINGS))
        s.arm()
        s.hold(0.1, **CLICK)
        s.hold(0.3)
        s.glide(0.5, (0.5, 0.5), (0.7, 0.6))
        s.hold(0.4, **TWO)
        s.glide(0.4, (0.5, 0.5), (0.5, 0.6), **TWO)
        return [repr(a) for a in s.actions]
    assert run() == run()
