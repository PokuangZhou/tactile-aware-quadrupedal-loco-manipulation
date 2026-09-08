#include <iostream>
#include <string>
#include <vector>
#include <unistd.h>
#include <termios.h>
#include <fcntl.h>
#include <errno.h>

#include <unitree/robot/channel/channel_factory.hpp>
#include <unitree/robot/b2/motion_switcher/motion_switcher_client.hpp>

using namespace unitree::robot;
using namespace unitree::robot::b2;

static struct termios g_oldt;

// Enable noncanonical mode and nonblocking keyboard input.
static void setKeyboardNonBlocking(bool enable) {
    if (enable) {
        struct termios newt;
        tcgetattr(STDIN_FILENO, &g_oldt);
        newt = g_oldt;
        newt.c_lflag &= ~(ICANON | ECHO);
        tcsetattr(STDIN_FILENO, TCSANOW, &newt);

        int flags = fcntl(STDIN_FILENO, F_GETFL, 0);
        fcntl(STDIN_FILENO, F_SETFL, flags | O_NONBLOCK);
    } else {
        tcsetattr(STDIN_FILENO, TCSANOW, &g_oldt);
        int flags = fcntl(STDIN_FILENO, F_GETFL, 0);
        fcntl(STDIN_FILENO, F_SETFL, flags & ~O_NONBLOCK);
    }
}

// Read one key without blocking; return false when no key is available.
static bool readKey(char& out) {
    char c;
    ssize_t n = read(STDIN_FILENO, &c, 1);
    if (n > 0) { out = c; return true; }
    return false;
}

// Retry CheckMode failures up to N times (for error 3104 timeouts, etc.).
static int checkModeWithRetry(MotionSwitcherClient& msc,
                             std::string& form,
                             std::string& motionName,
                             int max_retry = 5,
                             int sleep_ms = 300) {
    for (int i = 0; i < max_retry; i++) {
        form.clear();
        motionName.clear();
        int ret = msc.CheckMode(form, motionName);
        if (ret == 0) return 0;

        std::cout << "[CheckMode] failed ret=" << ret
                  << " (retry " << (i + 1) << "/" << max_retry << ")\n";
        usleep(sleep_ms * 1000);
    }
    // Make one final attempt and return its result.
    return msc.CheckMode(form, motionName);
}

// Print the current mode.
static bool printCurrentMode(MotionSwitcherClient& msc) {
    std::string form, name;
    int ret = checkModeWithRetry(msc, form, name);
    if (ret != 0) {
        std::cout << "[CheckMode] failed ret=" << ret << "\n";
        return false;
    }
    std::cout << "[Now   ] form=" << form << " motionName=" << name << "\n";
    return true;
}

// Release the current owner (for example, mcf).
static bool doRelease(MotionSwitcherClient& msc) {
    std::string form, name;
    int ret = checkModeWithRetry(msc, form, name);
    if (ret != 0) {
        std::cout << "[CheckMode] failed ret=" << ret << "\n";
        return false;
    }

    std::cout << "[Before] form=" << form << " motionName=" << name << "\n";

    int r2 = msc.ReleaseMode();
    std::cout << "[ReleaseMode] ret=" << r2 << "\n";
    usleep(200 * 1000);

    // Check once more.
    ret = checkModeWithRetry(msc, form, name);
    if (ret != 0) {
        std::cout << "[CheckMode] failed ret=" << ret << "\n";
        return false;
    }
    std::cout << "[After ] form=" << form << " motionName=" << name << "\n";

    if (name.empty()) {
        std::cout << "[OK] low-level ready (motionName empty)\n";
        return true;
    } else {
        std::cout << "[WARN] still occupied by: " << name << "\n";
        return false;
    }
}

// Enable via SelectMode, preferring mcf and then normal/ai/advanced.
// static bool doEnable(MotionSwitcherClient& msc,
//                      const std::vector<std::string>& candidates) {
//     // Try each candidate in order.
//     for (const auto& mode : candidates) {
//         int r = msc.SelectMode(mode);
//         std::cout << "[SelectMode \"" << mode << "\"] ret=" << r << "\n";

//         // A zero result indicates success. The mode can rarely change after a
//         // nonzero result, so always print the current mode.
//         usleep(200 * 1000);
//         printCurrentMode(msc);

//         if (r == 0) {
//             return true;
//         }

//         // Error 7004 usually means the mode name is unsupported; try the next one.
//         // Continue after other errors too; break here if different behavior is needed.
//     }
//     std::cout << "[WARN] enable failed for all candidates\n";
//     return false;
// }

static bool doEnable(MotionSwitcherClient& msc,
                     const std::vector<std::string>& candidates,
                     int max_try_each = 5,
                     int wait_ms = 300) {
    for (const auto& mode : candidates) {
        for (int k = 0; k < max_try_each; k++) {
            int r = msc.SelectMode(mode);
            std::cout << "[SelectMode \"" << mode << "\"] ret=" << r
                      << " (try " << (k + 1) << "/" << max_try_each << ")\n";

            usleep(wait_ms * 1000);

            // Check the current state; this also gives DDS time to stabilize.
            std::string form, name;
            int ret = checkModeWithRetry(msc, form, name);
            if (ret == 0) {
                std::cout << "[Now   ] form=" << form << " motionName=" << name << "\n";
            } else {
                std::cout << "[CheckMode] failed ret=" << ret << "\n";
            }

            // Success means SelectMode returned zero or CheckMode reports the target mode.
            if (r == 0 || (!name.empty() && name == mode)) {
                return true;
            }

            // Error 7004 means the mode name is unsupported; skip further retries.
            if (r == 7004) break;
        }
    }

    std::cout << "[WARN] enable failed for all candidates\n";
    return false;
}


int main(int argc, const char** argv) {
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " networkInterface\n";
        std::cout << "Keys: d=release(low-level), e=enable(high-level), q=quit\n";
        return -1;
    }

    ChannelFactory::Instance()->Init(0, argv[1]);

    MotionSwitcherClient msc;
    msc.SetTimeout(10.0f);
    msc.Init();
    usleep(600 * 1000);  // Allow time for DDS discovery.

    std::cout << "Boot: try release motion control (low-level ready)\n";
    doRelease(msc);

    std::cout << "\nKeys:\n"
              << "  d: disable high-level (ReleaseMode) -> low-level ready\n"
              << "  e: enable high-level (SelectMode)  -> try mcf then normal/ai/advanced\n"
              << "  q: quit\n"
              << "NOTE: When press e, you should STOP publishing rt/lowcmd in your policy process.\n\n";

    setKeyboardNonBlocking(true);

    bool running = true;
    while (running) {
        char c = 0;
        if (readKey(c)) {
            if (c == 'q') {
                running = false;
            } else if (c == 'd') {
                std::cout << "\n[d] ReleaseMode => low-level\n";
                doRelease(msc);
            } else if (c == 'e') {
                std::cout << "\n[e] SelectMode => enable high-level\n";
                // Prefer "mcf", the current high-level owner on this robot.
                // Also try normal/ai/advanced for compatibility with future firmware.
                doEnable(msc, {"mcf", "normal", "ai", "advanced"});
            } else if (c == '\n' || c == '\r') {
                // ignore
            } else {
                // For any other key, print the current state for debugging.
                std::cout << "\n[?] key=" << c << " -> current mode:\n";
                printCurrentMode(msc);
            }
        }
        usleep(20 * 1000); // 20ms
    }

    setKeyboardNonBlocking(false);
    std::cout << "\nExit.\n";
    return 0;
}
