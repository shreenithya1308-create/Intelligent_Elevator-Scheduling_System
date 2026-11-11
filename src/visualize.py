"""Professional Pygame dashboard for the trained elevator Q-learning agent."""
import sys
import pygame

from environment import ElevatorEnv
from agent import QLearningAgent

WIDTH, HEIGHT = 1360, 850
FPS = 30

# Palette
BG = (13, 19, 32)
PANEL = (23, 32, 49)
PANEL_ALT = (29, 41, 61)
BORDER = (48, 64, 88)
TEXT = (232, 239, 249)
MUTED = (151, 166, 188)
CYAN = (54, 205, 220)
BLUE = (63, 135, 235)
GREEN = (55, 205, 145)
ORANGE = (245, 164, 75)
RED = (238, 99, 105)
PURPLE = (166, 135, 245)


def draw_text(surface, font, text, pos, color=TEXT):
    surface.blit(font.render(str(text), True, color), pos)


def panel(surface, rect, title, fonts):
    pygame.draw.rect(surface, PANEL, rect, border_radius=15)
    pygame.draw.rect(surface, BORDER, rect, width=1, border_radius=15)
    draw_text(surface, fonts["small_bold"], title.upper(), (rect.x + 18, rect.y + 15), MUTED)


def draw_metric(surface, rect, label, value, accent, fonts, suffix=""):
    pygame.draw.rect(surface, PANEL_ALT, rect, border_radius=12)
    pygame.draw.rect(surface, BORDER, rect, width=1, border_radius=12)
    pygame.draw.rect(surface, accent, (rect.x, rect.y + 10, 4, rect.height - 20), border_radius=2)
    draw_text(surface, fonts["tiny_bold"], label.upper(), (rect.x + 14, rect.y + 10), MUTED)
    draw_text(surface, fonts["large_bold"], f"{value}{suffix}", (rect.x + 14, rect.y + 31), TEXT)


def main():
    try:
        agent = QLearningAgent()
        agent.load("q_table.npy")
    except FileNotFoundError:
        print("q_table.npy not found. Run: python train.py")
        return
    except ValueError as exc:
        print(exc)
        return

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("ElevateIQ | Intelligent Elevator Control")
    clock = pygame.time.Clock()
    fonts = {
        "title": pygame.font.SysFont("segoeui", 29, bold=True),
        "subtitle": pygame.font.SysFont("segoeui", 15),
        "section": pygame.font.SysFont("segoeui", 19, bold=True),
        "body": pygame.font.SysFont("segoeui", 17),
        "small": pygame.font.SysFont("segoeui", 14),
        "small_bold": pygame.font.SysFont("segoeui", 13, bold=True),
        "tiny_bold": pygame.font.SysFont("segoeui", 11, bold=True),
        "large_bold": pygame.font.SysFont("segoeui", 25, bold=True),
        "floor": pygame.font.SysFont("segoeui", 14, bold=True),
    }

    env = ElevatorEnv(max_steps=100000, arrival_probability=0.18)
    state = env.reset(seed=2026)
    paused = False
    speed = 1
    episode_reward = 0.0
    last_action = "READY"
    served_recent = 0
    completed_flash = 0

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_UP:
                    speed = min(12, speed + 1)
                elif event.key == pygame.K_DOWN:
                    speed = max(1, speed - 1)
                elif event.key == pygame.K_r:
                    state = env.reset(seed=None)
                    episode_reward = 0.0
                    last_action = "RESET"
                    served_recent = 0
                    completed_flash = 0
                elif event.key == pygame.K_ESCAPE:
                    running = False

        if not paused:
            for _ in range(speed):
                valid_actions = env.valid_actions()
                action = agent.choose_action(state, valid_actions, training=False)
                state, reward, done, info = env.step(action)
                episode_reward += reward
                last_action = {0: "MOVE UP", 1: "MOVE DOWN", 2: "OPEN / SERVE"}[action]
                if info["completed_this_step"]:
                    served_recent = info["completed_this_step"]
                    completed_flash = 12
                elif completed_flash > 0:
                    completed_flash -= 1
                if done:
                    state = env.reset(seed=None)
                    episode_reward = 0.0

        screen.fill(BG)
        # Header
        draw_text(screen, fonts["title"], "ElevateIQ", (30, 20), TEXT)
        draw_text(screen, fonts["subtitle"], "REINFORCEMENT LEARNING  /  Q-LEARNING CONTROL CENTER", (32, 58), MUTED)
        status_color = ORANGE if paused else GREEN
        pygame.draw.circle(screen, status_color, (1300, 39), 6)
        draw_text(screen, fonts["small_bold"], "PAUSED" if paused else "LIVE SIMULATION", (1130, 30), status_color)
        pygame.draw.line(screen, BORDER, (30, 85), (1330, 85), 1)

        # Top KPI cards
        kpi_y, kpi_h, kpi_w, gap = 105, 82, 245, 17
        kpis = [
            ("CURRENT FLOOR", f"{env.floor:02d} / 09", CYAN),
            ("PASSENGERS ONBOARD", f"{len(env.onboard)} / {env.capacity}", BLUE),
            ("JOURNEYS COMPLETED", env.total_served, GREEN),
            ("AVG WAIT TO BOARD", f"{env.metrics()['average_wait']:.1f} steps", ORANGE),
            ("CUMULATIVE REWARD", f"{episode_reward:.1f}", PURPLE),
        ]
        for i, (label, value, accent) in enumerate(kpis):
            draw_metric(screen, pygame.Rect(30 + i * (kpi_w + gap), kpi_y, kpi_w, kpi_h),
                        label, value, accent, fonts)

        # Main panels
        building_rect = pygame.Rect(30, 205, 600, 600)
        activity_rect = pygame.Rect(650, 205, 330, 600)
        insight_rect = pygame.Rect(1000, 205, 330, 600)
        panel(screen, building_rect, "Building overview · 10 floors", fonts)
        panel(screen, activity_rect, "Passenger activity", fonts)
        panel(screen, insight_rect, "Agent intelligence", fonts)

        # Building shaft
        shaft_x, shaft_y, shaft_w, floor_gap = 270, 260, 150, 48
        shaft_h = floor_gap * 10
        pygame.draw.rect(screen, (17, 26, 42), (shaft_x, shaft_y, shaft_w, shaft_h), border_radius=8)
        pygame.draw.rect(screen, BORDER, (shaft_x, shaft_y, shaft_w, shaft_h), width=2, border_radius=8)
        for floor in range(env.num_floors):
            y = shaft_y + (env.num_floors - 1 - floor) * floor_gap
            pygame.draw.line(screen, (65, 82, 108), (shaft_x, y + floor_gap - 2),
                             (shaft_x + shaft_w, y + floor_gap - 2), 1)
            draw_text(screen, fonts["floor"], f"FLOOR {floor:02d}", (65, y + 14), MUTED)
            queue_count = len(env.waiting[floor])
            # Call indicator
            dot_color = ORANGE if queue_count else (57, 75, 99)
            pygame.draw.circle(screen, dot_color, (465, y + 21), 6)
            draw_text(screen, fonts["small"], f"{queue_count:02d} waiting", (480, y + 12),
                      ORANGE if queue_count else MUTED)
            if queue_count:
                pygame.draw.circle(screen, (90, 61, 39), (465, y + 21), 10, 1)

        # Elevator car
        car_y = shaft_y + (env.num_floors - 1 - env.floor) * floor_gap + 5
        car_rect = pygame.Rect(shaft_x + 18, car_y, shaft_w - 36, 37)
        pygame.draw.rect(screen, BLUE, car_rect, border_radius=7)
        pygame.draw.rect(screen, (137, 190, 255), car_rect, width=1, border_radius=7)
        pygame.draw.line(screen, (190, 218, 255), (car_rect.x + 25, car_rect.y + 7),
                         (car_rect.x + 25, car_rect.bottom - 7), 2)
        pygame.draw.line(screen, (190, 218, 255), (car_rect.right - 25, car_rect.y + 7),
                         (car_rect.right - 25, car_rect.bottom - 7), 2)
        occupancy_text = fonts["small_bold"].render(f"{len(env.onboard)}/{env.capacity}", True, (255, 255, 255))
        screen.blit(occupancy_text, occupancy_text.get_rect(center=car_rect.center))

        # Building panel footer
        direction_text = "UPWARD" if env.direction > 0 else "DOWNWARD" if env.direction < 0 else "IDLE"
        draw_text(screen, fonts["small_bold"], f"DIRECTION  {direction_text}", (65, 755), CYAN)
        draw_text(screen, fonts["small"], f"Last decision: {last_action}", (280, 755), MUTED)

        # Activity panel: passengers waiting on each floor, and onboard list.
        draw_text(screen, fonts["small_bold"], "WAITING QUEUES", (670, 250), CYAN)
        y = 280
        nonempty = [(floor, q) for floor, q in enumerate(env.waiting) if q]
        if not nonempty:
            draw_text(screen, fonts["small"], "No passengers currently waiting", (670, y), MUTED)
        else:
            for floor, queue in sorted(nonempty, key=lambda item: item[0], reverse=True)[:8]:
                pygame.draw.rect(screen, PANEL_ALT, (670, y, 290, 31), border_radius=7)
                draw_text(screen, fonts["small_bold"], f"Floor {floor}", (681, y + 7), TEXT)
                draw_text(screen, fonts["small"], f"{len(queue)} passenger(s)", (775, y + 7), ORANGE)
                destinations = ", ".join(str(p["destination"]) for p in queue[:5])
                draw_text(screen, fonts["small"], f"→ {destinations}", (865, y + 7), MUTED)
                y += 37

        draw_text(screen, fonts["small_bold"], "ONBOARD DESTINATIONS", (670, 590), CYAN)
        y = 620
        if not env.onboard:
            draw_text(screen, fonts["small"], "Elevator is empty", (670, y), MUTED)
        else:
            for idx, passenger in enumerate(env.onboard[:6], start=1):
                pygame.draw.circle(screen, BLUE, (681, y + 9), 5)
                draw_text(screen, fonts["small"],
                          f"Passenger {idx}  →  Floor {passenger['destination']}",
                          (696, y), TEXT)
                y += 25

        # Agent insight panel
        draw_text(screen, fonts["small_bold"], "CURRENT OBSERVATION", (1020, 250), CYAN)
        state_labels = [
            ("Floor", state[0]), ("Direction code", state[1]),
            ("Waiting above (bucket)", state[2]), ("Waiting below (bucket)", state[3]),
            ("Onboard destinations above", state[4]), ("Onboard destinations below", state[5]),
            ("Waiting on this floor", state[6]), ("Drop-offs on this floor", state[7]),
            ("Passengers onboard", state[8]),
        ]
        y = 282
        for label, value in state_labels:
            draw_text(screen, fonts["small"], label, (1020, y), MUTED)
            draw_text(screen, fonts["small_bold"], value, (1280, y), TEXT)
            y += 27

        pygame.draw.line(screen, BORDER, (1020, 540), (1310, 540), 1)
        draw_text(screen, fonts["small_bold"], "SIMULATION METRICS", (1020, 558), CYAN)
        metrics = env.metrics()
        metric_lines = [
            ("Time step", env.time),
            ("Total floor movements", metrics["movement"]),
            ("Passengers arrived", metrics["arrivals"]),
            ("Waiting at present", metrics["total_waiting_people"]),
            ("Invalid actions", metrics["invalid_actions"]),
        ]
        y = 590
        for label, value in metric_lines:
            draw_text(screen, fonts["small"], label, (1020, y), MUTED)
            draw_text(screen, fonts["small_bold"], value, (1280, y), TEXT)
            y += 28

        # Footer controls
        pygame.draw.line(screen, BORDER, (30, 820), (1330, 820), 1)
        footer = "SPACE  Pause/Resume     ↑  Faster     ↓  Slower     R  Reset     ESC  Exit"
        draw_text(screen, fonts["small"], footer, (35, 825), MUTED)
        draw_text(screen, fonts["small"], f"Simulation speed: {speed}x", (1170, 825), CYAN)

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
