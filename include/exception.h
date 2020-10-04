#include <setjmp.h>

/// 异常标记
typedef struct tagExcepSign
{
    jmp_buf _StackInfo; // 保存异常处理入口的堆栈信息
    int _ExcepType; // 异常类型，0表示无异常，异常类型号一般取小于0的数
} ExcepSign;

/// 获取异常类型号
#define ExcepType(ExcepSign) ((ExcepSign)._ExcepType)

/// 可能抛出异常的代码块
#define Try(ExcepSign) if ( ((ExcepSign)._ExcepType = setjmp((ExcepSign)._StackInfo)) == 0 )

/// 捕获特定异常
#define Catch(ExcepSign, ExcepType) else if ((ExcepSign)._ExcepType == (ExcepType))

/// 捕获所有可能异常
#define CatchElse(ExcepSign) else if((ExcepSign)._ExcepType < 0)

/// 抛出异常
#define Throw(ExcepSign, ExcepType) longjmp((ExcepSign)._StackInfo, ExcepType)