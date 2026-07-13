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

// 进入非规范模式 + 非阻塞读取键盘
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

// 读取一个按键（非阻塞）；没有按键返回 false
static bool readKey(char& out) {
    char c;
    ssize_t n = read(STDIN_FILENO, &c, 1);
    if (n > 0) { out = c; return true; }
    return false;
}

// 稳一点的 CheckMode：失败重试 N 次（处理 3104 timeout 等）
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
    // 最后再做一次，返回最终码
    return msc.CheckMode(form, motionName);
}

// 打印当前 mode
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

// Release：释放当前占用者（例如 mcf）
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

    // 再检查一次
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

// Enable：尝试 SelectMode（优先 mcf，其次 normal/ai/advanced）
// static bool doEnable(MotionSwitcherClient& msc,
//                      const std::vector<std::string>& candidates) {
//     // 候选列表依次尝试
//     for (const auto& mode : candidates) {
//         int r = msc.SelectMode(mode);
//         std::cout << "[SelectMode \"" << mode << "\"] ret=" << r << "\n";

//         // r==0 视为成功；但也有可能 r!=0 时仍改变（少见），所以都打印一次当前 mode
//         usleep(200 * 1000);
//         printCurrentMode(msc);

//         if (r == 0) {
//             return true;
//         }

//         // 7004 通常是 mode name 不支持；继续尝试下一个
//         // 其他错误也继续尝试（你可以按需求改成遇到非7004就 break）
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

            // 看看当前状态（顺便能稳定 DDS）
            std::string form, name;
            int ret = checkModeWithRetry(msc, form, name);
            if (ret == 0) {
                std::cout << "[Now   ] form=" << form << " motionName=" << name << "\n";
            } else {
                std::cout << "[CheckMode] failed ret=" << ret << "\n";
            }

            // 成功条件：SelectMode ret==0，或者 CheckMode 已经变成目标 mode
            if (r == 0 || (!name.empty() && name == mode)) {
                return true;
            }

            // 7004=名字不支持：直接换下一个 mode，别重试浪费时间
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
    usleep(600 * 1000);  // 给 DDS discovery 一点时间

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
                // 你这台机子高层占用者是 mcf，所以优先尝试 "mcf"
                // 如果未来固件变化，也会尝试 normal/ai/advanced
                doEnable(msc, {"mcf", "normal", "ai", "advanced"});
            } else if (c == '\n' || c == '\r') {
                // ignore
            } else {
                // 其他键：打印一次当前状态，方便调试
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
