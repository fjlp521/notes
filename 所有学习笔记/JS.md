#### 1、new 一个对象的过程

```js
function Mother(lastname){
    this.lastname = lastname
}
let son = new Mother("Da")
```

- 创建一个新对象：son

- 新对象会被执行 [[prototype]] 连接：son.\__proto__ = Mother.prototype

- 新对象和函数调用的 this 会绑定起来：Mother.call(son, "Da")

- 执行构造函数中的代码：son.lastname

- 如果函数没有返回值，那么就会自动返回这个新对象

  ```js
  //函数没有返回值，需要用new接
  function func1(lastname) {
      this.lastname = lastname
  }
  //函数有返回值，正常调用即可
  function func2(lastname) {
      return {
          lastname    //对象简写形式
      }
  }
  let obj1 = new func1('lastname')
  let obj2 = func2('lastname')
  console.log(obj1)   //func1 { lastname: 'lastname' }
  console.log(obj2)   //{ lastname: 'lastname' }
  ```

#### 2、DOM事件模型

###### 事件流

- 三个阶段：事件捕获阶段（从上至下）、处于目标阶段、事件冒泡阶段（从下至上）

- 事件执行顺序分析发生在：父子元素同时绑定了同一类型的事件（如点击），没有绑定同一类型事件你分析个屁

- 例子：1->2->3->4，绑定的事件类型分别为冒泡、捕获、捕获、冒泡，点击4后，事件执行顺序为2->3->4->1

###### 事件模型

- 三种：原始事件模型（DOM0）、标准事件模型（DOM2级）、IE事件模型（基本不用）

- 原始事件模型

  - HTML代码中绑定

    ```js
    <input type="button" onclick="fun()">
    ```

  - JS代码中绑定

    ```js
    var btn = document.getElementById('.btn');
    btn.onclick = fun;
    ```

  - 特性

    - 绑定速度快：有时绑定速度太快，可能页面还未完全加载出来，以至于事件可能无法正常运行
    - 只支持冒泡，不支持捕获
    - 同一个类型的事件只能绑定一次

  - 删除 DOM0 级事件处理程序只要将对应事件属性置为null即可

    ```js
    btn.onclick = null;
    ```

- 标准事件模型

  - 会完整的经历捕获、处理、冒泡三个过程

  - 事件绑定和移除

    ```js
    let button = document.getElementById('clickMe')
    //绑定，第三个参数表示是否在捕获阶段触发，默认为false，即在冒泡阶段触发
    button.addEventListener(type, listener, useCapture)
    //移除
    button.removeEventListener(type, listener, useCapture)
    ```

  - 特性
    - 可以在一个DOM元素上绑定多个事件处理器，各自并不会冲突
    - 当第三个参数(useCapture)设置为 true 就在捕获过程中执行，反之在冒泡过程中执行处理函数

###### 使用场景

- 为什么要阻止冒泡：举例，有一个消息框，内有一个删除按钮，点击按钮删除本条信息，点击其他地方显示消息详情，如果不阻止冒泡，点击删除按钮，会执行删除以及显示消息详情两个操作，显然不合理；**总结**：对于父子元素同时绑定同一类型的事件，且触发事件时会执行不同的操作，需对子元素的事件冒泡阻止

#### 3、var、let、const

- var：在全局作用域声明会直接挂到window上，在局部作用域不会；有变量提升，可重复声明，因为这两点有时会带来意外效果

  变量提升：

  ```js
  console.log(a) // undefined
  var a = 20
  
  //以上代码编译后，按如下执行
  var a
  console.log(a)	// undefined
  a = 20
  ```

  变量提升+重复声明

  ```js
  var a = 1 
  function scope(){
      console.log(a)	//这里的a是经过变量提升的重复声明的a
      var a = 2
  }
  scope()	//输出undefined
  ```

- let、const ：块级作用域，无变量提升、不可重复声明等

- 注意：如果什么都不写，直接 a = 10（不论写在全局作用域还是局部作用域），则这个a会直接挂在window上

  ```js
  function test() {
      a = 10
      console.log(a)  
  }
  test()  //10
  console.log(a)  //10
  ```

#### 4、原型和原型链

```js
let a = [1,2,3]
console.log(a.__proto__ === Array.prototype)	//true
console.log(Object.prototype.__proto__)			//null
```

- \__proto__等同于[[prototype]]，表示该对象继承的东西（也是一个对象）

- prototype表示该对象留给孩子的东西

- 孩子继承父亲的，父亲留给孩子的，其实指向同一块东西（同一个对象），只不过\__proto__是孩子视角，prototype是父亲视角

- 任何对象都有\__proto__属性，但只有（构造）函数或者类有prototype属性（既有能力生成对象）

- 特殊的：Object.prototype.\__proto__ === null

- 关于构造函数：js中的函数从出生就自带一个prototype的属性，prototype从出生就自带一个constructor，且f.prototype.constructor === f，**注意：箭头函数没有prototype**

- Object 和 Function 的关系

  ```js
  Object instanceof Function 			//true
  Function instanceof Object			//true
  //原因如下：
  Object.prototype.constructor.__proto__ === Function.prototype	//解释：Object.prototype.constructor是一个函数
  Function.prototype.__proto__ === Object.prototype				//解释：Function.prototype是一个对象
  ```

#### 5、JS中的 this 绑定规则

1. this 四个绑定规则

   - 默认绑定，独立函数调用（包括多重嵌套但本质还是独立调用的情况）

   - 隐式绑定，由某一对象进行函数调用

   - 显式绑定，call、apply、bind（注意：bind使用过后，this就会固定，不受其他方式影响）

   - new 绑定

```js
function sum(x, y) {
    console.log(this, x + y)
}

const obj = {
    name: 'curry',
    age: 30,
    sum
}

sum(10, 20)                 	//默认绑定，输出为：window 30
obj.sum(10, 30)             	//隐式绑定，输出为：obj 30
sum.call(obj, 10, 20)       	//显示绑定call，输出为：obj 30
sum.apply(obj, [10, 20])    	//显示绑定apply，输出为：obj 30

const bindObj = sum.bind(obj) 	//bind返回一个函数
bindObj() 						//显示绑定bind，输出：obj 30
const obj2 = {}
bindObj.call(obj2)				//this不受影响，仍然指向obj
const newObj = new sum(10, 20)	//new绑定，输出：sum {} 30
```

2. JS 内置函数的this绑定
   - setTimeout（回调为普通函数）：默认绑定，独立调用，this为window
   - DOM事件监听：隐式绑定，this指向当前DOM对象，```box.onclick = function(){}```
   - 数组内置方法forEach、map等方法的第二个参数可以自由设置this，```forEach(callbackFn, thisArg)```
3. this绑定规则优先级
   - new绑定 > 显示绑定（apply/call/bind） > 隐式绑定 > 默认绑定
   - 注意new关键字不能和apply、call一起使用，所以不太好进行比较，默认为new绑定是优先级最高的
4. node 环境下全局this指向为 {}，即空对象
   - 当 js 文件被 node 执行时，该  js 文件会被视为一个模块
   - node 加载编译这个模块，将模块中的所有代码放入到一个函数中
   - 然后会执行这个函数，在执行这个函数时会通过apply绑定一个this，而绑定的this就为`{}`

#### 6、普通函数和箭头函数的 this 

> 普通函数的this在**调用时**确定，箭头函数的this在**定义时**确定，确定箭头函数this方法：向外层作用域中，一层层查找this ，直到有this 的定义，注意这里的作用域指处于函数内部，处于对象内部不算

```js
const obj1 = {
    obj2: {
        show: () => {
            console.log(this)
        }
    }
}
obj1.obj2.show()	//仍然还是window

const obj1 = {
    obj2: {
        show() {
            //此处可以找到this的定义
            return () => {
                console.log(this)	//这里的this和show()内部的this相同
            }
        }
    }
}
obj1.obj2.show()()	//obj2
```

#### 7、异步

- 异步并不是多线程，JS世界里只有单线程，异步只是通过调整任务执行顺序达到提高效率的目的。将某些任务暂时挂起（放到任务队列中），先执行后面的任务，之后再回过头执行那些挂起的任务，所谓的回过头就是“事件循环”机制
- 异步可以提升效率的原因：以网络请求为例，网络请求大体分为两部分，一是发起请求，二是等待结果。同步执行为：发起请求，等待结果，执行其余任务；异步执行为：发起请求，然后将该任务挂起，执行其余任务（同时等待结果）。可以看到异步执行将等待的时间用于执行其余任务，因此提升了效率（注意：等待是等待服务器，自身不消耗资源，因此可以不放在主线程）。
- 异步实现：宏任务（setTimeout等）和微任务（MutationObserver等）
- 异步和多线程的机制其实类似：假设程序顺序执行，中间会存在很多空余时间（等待时间），而异步和多线程都是通过调整程序执行顺序，将顺序执行时的空余时间用来工作，以达到提高效率的目的

#### 8、ES6+新特性

- let/const

- 数组和对象的解构，数组用[]，对象用{}，可以使用逗号省略，可以使用...，可以使用默认值

- 模板字符串、带标签的模板字符串

- ...展开运算符，写在变量前面，...arr

- 箭头函数

- 对象字面量增强：一是属性名和属性值相同时可以只写属性名；二是属性名可计算得来，使用[]，括号内写语句

- Proxy、Reflect：Proxy设置代理，Reflect用于提供对于对象属性增删改查的统一接口

- class、静态方法、extends继承

- Symbol：用于给对象定义一个独一无二的属性名，可用于定义私有成员，给对象的toString、Iterator等提供标识

- 迭代器、for...of

  ```js
  const obj = {
      life: [1,2,3,4],
      work: ['one', 'two', 'three', 'four'],
      [Symbol.iterator]: function() {
          const arr = [...this.life, ...this.work]
          let i = 0, n = arr.length
          return {
              next: function() {
                  return {
                      value: arr[i],
                      done: i++ >= n
                  }
              }
          }
      }
  }
  for(i of obj) {
      console.log(i)
  }
  ```

- 生成器

  ```js
  //基本使用
  function * foo() {
      console.log('111')
      yield 111
      console.log('222')
      yield 222
      console.log('333')
      yield 333
  }
  const generator = foo()
  console.log(generator.next())
  console.log(generator.next())
  console.log(generator.next())
  console.log(generator.next())
  
  //生成器实现iterator
  const obj = {
      life: [1,2,3,4],
      work: ['one', 'two', 'three', 'four'],
      [Symbol.iterator]: function * () {
          const arr = [...this.life, ...this.work]
          for(i in arr) {
              yield arr[i]
          }
      }
  }
  for(i of obj) {
      console.log(i)
  }
  ```

- promise、async、await

#### 9、函数式编程

> 这里的函数指的是一种映射关系，等同于数学中的函数，而非定义一个方法。
>

##### 函数是一等公民

- 函数可以存储在变量中
- 函数可以作为参数
- 函数可以作为返回值

##### 高阶函数

- 可以将函数作为参数传递给另一个函数

  ```js
  function filter(array, fn) {
      let result = []
      for (const item of array) {
          if (fn(item)) {
              result.push(item)
          }
      }
      return result
  }
  ```

- 可以将函数作为另一个函数的返回结果

  ```js
  //fn函数只会执行一次
  function once(fn){
      let done = false
      return function(){
          if(!done) {
              done = true
              fn(...arguments)
          }
      }
  }
  
  let pay = once(function(money){
      console.log(`支付了 ${money} RMB`)
  })
  pay(5)	//fn会执行
  pay(5)	//fn不会执行
  pay(5)	//fn不会执行
  ```

##### 闭包

- 概念：函数和其周围的状态（词法环境）的引用捆绑在一起形成闭包

  - 可以在另一个作用域中调用一个函数的内部函数并访问到该函数的作用域中的成员，即延长了作用域

    ```js
    function makeFn() {
        let msg = "hello"
        return function() {
            console.log(msg)
        }
    }
    makeFn()()
    ```

  - 闭包的本质：函数在执行的时候会放到执行栈上，当函数执行完毕后会从执行栈上移除，**但是堆上的作用域成员因为被外部引用不能释放**，因此还可以被访问

- 闭包案例

  ```js
  function makePower(power) {
      return function(number) {
          return Math.pow(number, power)
      }
  }
  
  let power2 = makePower(2)   //生成一个求2次方的函数
  let power3 = makePower(3)   //生成一个求3次方的函数
  ```

##### 纯函数

> 函数式编程中的函数说的就是纯函数
>

- 概念：**相同的输入永远会得到相同的输出**，而且没有任何可观察的副作用，类似于数学中的 y = f(x)
- 数组中的 slice() 不会改变原数组，是纯函数，而 splice() 函数会修改原数组，是不纯的函数

##### 柯里化

###### 概念

- 当一个函数有多个参数的时候先传递一部分参数调用它（这部分参数以后永远不变）
- 然后返回一个新的函数接收剩余的参数，返回结果

###### 案例一

```js
function checkAge(min) {
    return function (age) {
        return age >= min
    }
}
let checkAge18 = checkAge(18)
let checkAge20 = checkAge(20)
//ES6
let checkAge = min => (age => age >= min)
console.log(checkAge(18)(20))
```

###### 案例二

```js
function match(reg) {
    return function(str) {
        return str.match(reg)
    }
}
let haveSpace = match(/\s+/g)   //生成一个空格匹配器
let haveNumber = match(/\d+/g)  //生成一个数字匹配器

function filter(func) {
    return function(array) {
        return array.filter(func)
    }
}
let findSpace = filter(haveSpace)   //过滤数组中含有空格的元素
console.log(filter(haveSpace)(['hell o', 'hhh']))
console.log(findSpace(['h l']))
```

###### 扩展

- 可以使用 lodash 库中的 curry() 方法将一个函数改造成柯里化函数

  ```js
  const _ = require('lodash')
  function getSum(a, b, c) {
      return a + b + c
  }
  let newSum = _.curry(getSum)
  console.log(newSum(1,2,3))
  console.log(newSum(1,2)(3))
  console.log(newSum(1)(2,3))
  console.log(newSum()(1,2,3))
  console.log(newSum(1)(2)(3))
  ```

- 案例二中的 match 和 filter 都可以用 curry() 改造

- curry() 方法的实现

  ```js
  function curry(func) {
      return function curriedFn(...args) {
          //判断实参和形参的个数
          if (args.length < func.length) {
              return function() {		
                  //concat用于合并两个数组，Array.from()将arguments转化为数组
                  return curriedFn(...args.concat(Array.from(arguments)))
              }
          }
          return func(...args)
      }
  }
  ```

- ...args 和 arguments

  ```js
  //args收集未赋值的参数到一个数组中，arguments是一个对象，保存了所有参数
  function test(a,b,...args) {
      console.log(args)   //[ 3, 4 ]
      console.log(arguments)  //[Arguments] { '0': 1, '1': 2, '2': 3, '3': 4 }
  }
  test(1,2,3,4)
  ```

###### 总结

- 柯里化可以让我们给一个函数传递较少的参数得到一个已经记住了某些固定参数的新函数
- 这是一种对函数参数的“缓存”
- 让函数变得更灵活，让函数的粒度更小
- 可以把多元函数转化为一元函数，可以组合使用函数产生强大的功能
- 自己的话理解柯里化：柯里化就类似于半成品，相对于原始材料更加集中，相对于成品更加灵活

##### 函数组合

> 将洋葱式函数调用转化为链式调用：f(g(h(x))) -> fn(f, g, h), fn(x)

```js
function test(f, g) {
    return function(x) {
        return f(g(x))
    }
}
```

- lodash 中 flow 和 flowRight 用于函数组合，一个从左往右，一个从右往左

  ```js
  const _ = require('lodash')
  let newFunc = _.flowRight(f, g)
  ```

  实现原理

  ```js
  function compose(...args) {
      return function(initValue) {
          return args.reverse().reduce(function(acc, fn){
              return fn(acc)
          }, initValue)
      }
  }
  //箭头函数
  let compose = (...args) => value => args.reverse().reduce((acc, fn) => fn(acc), value)
  ```

- 函数组合满足结合律（前提是函数顺序一致）

- 函数组合要求函数只有一个参数，对于多个参数的函数，先进行柯里化，改造为一元函数，之后再进行组合；lodash 中的 fp 模块中封装有柯里化后的函数，这些函数参数特点是函数在前，数据在后

  ```js
  const _ = require('lodash')
  const fp = require('lodash/fp')
  const f = fp.flowRight(fp.join('-'), fp.map(_.toLower), fp.split(' '))
  ```

##### Functor（函子）

```js
class Container {
    constructor(value) {
        this._value = value
    }
    map(fn) {
        //返回一个新的对象
        return new Container(fn(this._value))
    }
}
let r = new Container(5).map(x => x+1).map(x => x*x)	//链式调用
```

- MayBe函子，解决传入的值为null的情况

  ```js
  class MayBe {
      //静态方法，避免每次写new MayBe()
      static of(value) {
          return new MayBe(value)
      }
      constructor(value) {
          this._value = value
      }
      map(fn) {
          return this.isNothing() ? MayBe.of(null) : MayBe.of(fn(this._value))
      }
      isNothing() {
          return this._value === null || this._value === undefined
      }
  }
  let r = MayBe.of(null).map(x => x + 1)	//不会报异常
  ```

- Either函子，MayBe可以解决空值问题，但无法提示哪一步出现了空值

  ```js
  class Left {
      static of(value) {
          return new Left(value)
      }
      constructor(value) {
          this._value = value
      }
      map(fn) {	//区别在这个地方
          return this
      }
  }
  //Right就是普通的函子
  class Right {
      static of(value) {
          return new Right(value)
      }
      constructor(value) {
          this._value = value
      }
      map(fn) {
          return Right.of(fn(this._value))
      }
  }
  function parseJSON(str) {
      try {
          return Right.of(JSON.parse(str))        
      } catch (e) {
          return Left.of({error: e.message})
      }
  }
  ```

- IO函子

  - IO函子中的 _value 是一个函数，这里是把函数作为值来处理
  - IO函子可以把不纯的动作存储到 _value 中，延迟执行这个不纯的动作（惰性执行），包装当前的纯操作
  - 把不纯的操作交给调用者来处理

  ```js
  //代码没看明白！！！
  const fp = require('lodash/fp')
  class IO {
      static of(value) {
          return new IO(() => value)
      }
      constructor(fn) {
          this._value = fn
      }
      map(fn) {
          return new IO(fp.flowRight(fn, this._value))
      }
  }
  //process是node中的对象
  let r = IO.of(process).map(p => p.execPath)
  console.log(r._value())
  ```

- Task函子
- Monad函子

#### 10、Promise使用
##### Promise 基本使用
```js
function ajax(url) {
    return new Promise(function(resolve, reject){
        let xhr = new XMLHttpRequest()
        xhr.open('GET', url)
        xhr.onload = function() {
            if (this.status === 200) {
                resolve(this.response)
            }else{
                reject(this.statusText)
            }
        }
        xhr.send()
    })
}
ajax("http://httpbin.org/get").then(
    function(res){      //执行成功之后的回调
        console.log(res) 
    },
    function(error){    //执行失败之后的回调
        console.log(error) 
    }
)
```
- new Promise() 中传入一个执行函数，该函数接收两个参数，resolve和reject
- then() 中传入两个函数，分别为 执行成功的回调函数 和 执行失败的回调函数
- catch() 中传入一个函数，为 执行失败的回调函数，catch() 是 then(null, failureCallback) 的简写形式
- then() 中使用失败回调和 catch() 中使用失败回调的区别：then() 只能捕获当前 Promise 对象中发生的异常，而 catch() 可以捕获整个 Promise 链条上发生的异常，故推荐使用 catch()

##### Promise 链式调用

- then() 会返回一个全新的 Promise 对象
- 后面的 then() 就是在为上一个 then 返回的 Promise 注册回调
- 前面 then() 中的回调函数的返回值会作为后面 then() 回调的参数
- 如果回调中返回的是 Promise，那后面 then() 的回调会等待它的结果

##### async/await

解决多个Promise链条可读性差的问题，使得异步代码的写法类似与同步代码

```js
//Promise链条，可读性差
ajax('1')
.then(value => ajax('2'))
.then(value => ajax('3'))
.then(value => ajax('4'))
.catch(error => { console.log(error) })
//使用类似与同步代码的方式书写异步代码
async function test() {
    try {
        let res1 = await ajax('1')	//目前await只能用于async函数内部
        let res2 = await ajax('2')
        let res3 = await ajax('3')
        let res4 = await ajax('4')
    } catch (error) {
        console.log(error)
    }
}
//并且async函数会返回一个Promise对象
let promise = test()
promise.then(() => {
    console.log("all completed")
})
```

#### 11、Promise剖析

###### 原理

- new Promise()时传入的是一个执行函数，resolve和reject是该函数的两个形参，new Promise()时会立刻调用执行函数，调用函数时的实参resolve和reject是Promise类内部定义好的两个函数。整个过程为：传入一个执行函数，在该函数中执行某些操作，根据执行结果调用resolve或者reject，注意，只管调用，不用管这两个函数的内部细节，相当于你留了两个坑，Promise调用执行函数的时候，会自动填上这两个坑，可以理解为这两个坑是做一些“收尾工作”

- 三种状态：待定、完成、拒绝，初始为待定，执行resolve时将待定转为完成，执行rejected时将待定转为拒绝，而且，只有这两种转换方式。使用状态的目的是：根据状态判断then到底该执行哪个函数

- 当执行函数为异步函数时，待其执行完之后才能执行then，处理方法：将then中传入的回调函数先保存下来，然后在resolve或者reject中调用

- 官方规范指出，then中的onFulfilled和onRejected应该异步执行，可采用setTimeout实现

- then可以多次调用，因此用数组来存储then中的函数，等待执行函数完成，会依次调用保存的函数

- then方法返回一个Promise对象

###### 基础版（没有链式调用）


```js
const PENDING = 'pending'
const FULFILLED = 'fulfilled'
const REJECTED = 'rejected'

class MyPromise {
    constructor(excutor) {
        this.state = PENDING        //初始为待定状态
        this.value = undefined      //保存resolve中的值，传给then用
        this.reason = undefined     //reject，传给then用
        this.onFulfilledCallbacks = []  //执行函数为异步时，保存then中的成功回调，等待执行函数结束后再执行
        this.onRejectedCallbacks = []	//执行函数为异步时，保存then中的失败回调，等待执行函数结束后再执行	
        const resolve = value => { 
            if (this.state === PENDING) {
                this.state = FULFILLED
                this.value = value
                console.log(value)
                if (this.onFulfilledCallbacks) {	//调用then中保存的成功回调
                    this.onFulfilledCallbacks.forEach(onFulfilled => onFulfilled(value))
                }
            }
        }
        const reject = reason => {
            if (this.state === PENDING) {
                this.state = REJECTED
                this.reason = reason
                console.log(reason)
                if (this.onRejectedCallbacks) {		//调用then中保存的失败回调
                    this.onRejectedCallbacks.forEach(onRejected => onRejected(reason))
                }
            }
        }
        try {
            excutor(resolve, reject)
        } catch (error) {
            throw error
        }
    }
    //当状态为FULFILLED或者REJECTED时，直接执行就可以； -> 对应执行函数为同步函数
    //当状态为PADDING时，说明前面异步函数未执行完成，先将函数保存起来   -> 对应执行函数为异步函数
    then(onFulfilled, onRejected) {
        if (this.state === FULFILLED) {    //执行函数为同步函数，立即执行成功回调
            typeof(onFulfilled) === 'function' && setTimeout(() => {	//短路简化写法
                onFulfilled(this.value)
            }, 0);
        }else if (this.state === REJECTED) {	//执行函数为异步函数，立即执行失败回调
            typeof(onRejected) === 'function' && setTimeout(() => {
                onRejected(this.reason)
            }, 0);
        }else{  
            if (typeof(onFulfilled) === 'function') {
                this.onFulfilledCallbacks.push( value => setTimeout(() => {
                    onFulfilled(value)
                }, 0));
            }
            if (typeof(onRejected) === 'function') {
                this.onRejectedCallbacks.push( reason => setTimeout(() => {
                    onRejected(reason)
                }, 0));
            }
        }
    }
}
```

###### 完整版（有链式调用以及值穿透）

```js
//略复杂，暂略
```

#### 12、三种方法实现：1秒后打印1，2秒后打印2，3秒后打印3

```js
//方法一：原始的setTimeout，存在回调地狱问题
function Test1() {
    setTimeout(() => {
        console.log('Test1：1秒后打印1')
        setTimeout(() => {
            console.log('Test1：2秒后打印2')
            setTimeout(() => {
                console.log('Test1：3秒后打印3')
            }, 3000);
        }, 2000);
    }, 1000);
}
Test1()

//方法二：promise，解决了回调地域，但调用时可读性较差
function Test2(time, value, testName='test2') {
    return new Promise((resolve, reject) => {
        setTimeout(() => {
            console.log(`${testName}：${time}秒后打印${value}`)
            resolve()	//表示已完成，不能不写
        }, time*1000);
    })
}
Test2(1, 1).then(()=>{Test2(2, 2).then(()=>{Test2(3,3)})})	//then里面要传一个新函数并包裹着Test2()，不能直接传Test2()，且注意，不是.then().then()，而是.then(.then())

//方法三：async/await，解决了promise可读性差的问题，如同步代码一样书写异步代码
async function Test3() {
    await Test2(1, 1, 'test3')
    await Test2(2, 2, 'test3')
    await Test2(3, 3, 'test3')
}
Test3()
```

#### 13、宏任务与微任务

- 常见宏任务：I/O 、setTimeout、setInterval；微任务：Promise.then、catch、finally、process.nextTick

- 在挂起任务时，JS 引擎会将 **所有任务** 按照类别分到这两个队列中，首先在 macrotask 的队列（这个队列也被叫做 task queue）中取出第一个任务，执行完毕后取出 microtask 队列中的所有任务顺序执行；之后再取 macrotask 任务，周而复始，直至两个队列的任务都取完
- 事件循环，每一次循环称为tick，每一次tick的任务如下：取出宏任务队列中的第一个任务并执行，然后执行微任务队列中的所有任务直至清空，接着开启下一轮tick：宏任务first - 微任务all - 宏任务first - 微任务all...
- 程序刚启动时为主线程，这时为宏任务，且立即执行该任务
- 可简单认为宏任务是由宿主发起（浏览器、node），微任务由 js 内部代码发起

案例一：

```js
setTimeout(() => {
    console.log('3')
}, 0)
console.log('1');

new Promise((resolve) => {	//promise中传入的执行函数也是立即执行的
    console.log('1.1');
    resolve()
}).then(() => {				//then为微任务
    console.log('2');
}).then(()=>{
    console.log('2.1')
})
//主线程宏任务：1 1.1
//宏任务：3
//微任务：2 2.1
//执行顺序：主线程（外层宏） -  微  -  宏
//输出结果：1 1.1 2 2.1 3
```
案例二：
```js
setTimeout(() => {
    console.log("宏2");
}, 1000);

setTimeout(() => {
    console.log("宏3");
}, 100);

console.log("同步");

new Promise((resolve, reject) => {
        setTimeout(() => {
            console.log("宏1");
        }, 0)
        console.log("立即");
        resolve();
        // reject()
    })
    .then(() => {
        console.log("微0");
    })
    .then(() => {
        console.log("微1");
    })
    .catch(() => {
        console.log("err");
    })
//执行结果：同步 立即 微0 微1 err 宏1 宏3 宏2
```

案例三：

```js
console.log('1');
setTimeout(function() {
    console.log('3');
    new Promise(function(resolve) {
        console.log('3.1');
        resolve();
    }).then(function() {
        console.log('4')
    })
})

new Promise(function(resolve) {
    console.log('1.1');
    resolve();
}).then(function() {
    console.log('2')
})

setTimeout(function() {
    console.log('5');
    new Promise(function(resolve) {
        console.log('5.1');
        resolve();
    }).then(function() {
        console.log('6')
    })
})
//执行结果：1 1.1 2 3.1 4 5 5.1 6
```

#### 14、三种属性遍历

- Object.keys()、Object.getOwnPropertyNames() 、Reflect.ownKeys()

- 前者只能遍历可枚举的属性，中者能遍历可枚举和不可枚举的属性，后者能遍历可枚举、不可枚举以及Symbol的属性

#### 15、JS手写面试题

###### 1、实现 instanceof

```js
const a = []
const b = {}
function myInstanceof(a, b) {
    while(a.__proto__ != null && a.__proto__ != b.prototype) {
        a = a.__proto__
    }
    return a.__proto__ != null
}
console.log(myInstanceof(a, Array))		//true
console.log(myInstanceof(a, Object))	//true
console.log(myInstanceof(b, Array))		//false
console.log(myInstanceof(b, Object))	//true
```

###### 2、实现 Array.map()

```js
const arr = [1,2,3,4]
Array.prototype.myMap = function(callbackfn, thisArg) {
    // thisArg = thisArg || []
    let arr = this, res = []
    for(let  i = 0; i < arr.length; i++) {
        res.push(callbackfn.call(thisArg, arr[i], i, arr))
    }
    return res
}
console.log(arr.myMap(x => x*x))
```

###### 3、实现 Array.reduce()

```js
const arr = [1,2,3]
//原始的reduce对于传入的initValue为null或者undefined时的处理方式不明，这里暂时简单处理
Array.prototype.myReduce = function(callbackfn, initValue) {
    if(!initValue && this.length == 0) {    //无初始值且数组为空
        throw new TypeError('Reduce of empty array with no initial value')
    }
    let arr = this
    let res = initValue || arr[0]	
    let i = initValue ? 0 : 1		//有initValue从0开始，没有时从1开始
    for(; i < arr.length; i++) {
        res = callbackfn(res, arr[i], i, arr)
    }
    return res
}
console.log(arr.myReduce((x, y) => x*y, 2))
```

###### 4、用 reduce 实现 map

```js
const arr = [1,2,3]
Array.prototype.mapUsingReduce = function(callbackfn, thisArg) {
    let arr = this
    let res = []
    arr.reduce(function(_, curValue, curIndex, array){
        res.push(callbackfn.call(thisArg, curValue, curIndex, array))
    }, [])
    return res
}
console.log(arr.mapUsingReduce(x => x*x))
```

###### 5、数组扁平化

基本使用
```js
let a = [1,2,[3, 4, [5, 6, [7, 8]]]]
console.log(a.flat())           //1、2、3、4、[5、6、[7、8]]
console.log(a.flat(2))          //1、2、3、4、5、6、[7、8]
console.log(a.flat(Infinity))   //1、2、3、4、5、6、7、8]
```

实现

```js
let a = [1,2,[3, 4, [5, 6, [7, 8]]]]
function myFlat(arr, deep) {
    let res = []
    for(item of arr) {
        if(Array.isArray(item) && deep) {
            res.push(...myFlat(item, deep-1))
        }else{
            res.push(item)
        }
    }
    return res
}
Array.prototype.myFlat = function(deep = 1) {
    return myFlat(this, deep)
}
console.log(a.myFlat())           //1、2、3、4、[5、6、[7、8]]
console.log(a.myFlat(2))          //1、2、3、4、5、6、[7、8]
console.log(a.myFlat(Infinity))   //1、2、3、4、5、6、7、8]
```

###### 6、柯里化

```js
function curry(func) {
    return function curriedFn(...args) {
        //判断实参和形参的个数
        if (args.length < func.length) {
            return function() {		//注意这里要返回一个可以生成柯里化的函数，而不能将curriedFn直接返回
                //concat用于合并两个数组，Array.from()将arguments转化为数组
                return curriedFn(...args.concat(Array.from(arguments)))
                // return curried(...[...args, ...Array.from(arguments)])
            }
        }
        return func(...args)
    }
}
function test(a, b, c, d) {
    return a + b + c + d
}
let a = curry(test)
console.log(a(1)(2)(3,4))	//10
```

###### 7、浅拷贝实现

```js
//1
function shallowCopy(target, source) {
    Reflect.ownKeys(target).forEach((item) => {	//Reflect.ownKeys可以获取所有属性（包括Symbol）
        source[item] = target[item]
    })
}
//2
Object.assign(target, source)

let a = [1,2,3,4]
//3
let b = a.slice()
//4
let c = a.concat()
//5
let d = [...a]
```

###### 8、深拷贝实现

```js
//1，流氓啊，只能拷贝可枚举的属性
let b = JSON.parse(JSON.stringfy(a))

//2、暂时还不完善
function deepClone(obj, hash = new WeakMap()) {
    if (obj == null) return obj; // 如果是null或者undefined我就不进行拷贝操作(null == undefined)
    if (obj instanceof Date) return new Date(obj);
    if (obj instanceof RegExp) return new RegExp(obj);
    // 可能是对象或者普通的值  如果是函数的话是不需要深拷贝
    if (typeof obj !== "object") return obj;
    // 是对象的话就要进行深拷贝
    if (hash.get(obj)) return hash.get(obj);		//使用hash是为了解决Object中存在环形引用的问题
    let cloneObj = new obj.constructor();
    // 找到的是所属类原型上的constructor,而原型上的 constructor指向的是当前类本身
    hash.set(obj, cloneObj);
    for (let key in obj) {
        if (obj.hasOwnProperty(key)) {
            // 实现一个递归拷贝
            cloneObj[key] = deepClone(obj[key], hash);
        }
    }
    return cloneObj;
}
```

###### 9、手写 call / apply / bind

call

```js
Function.prototype.myCall = function(thisArg){
    if(typeof this != 'function') {
        throw new Error('不是函数')
    }
    const obj = thisArg || window           //声明一个指向thisArg的对象，也可直接thisArg = thisArg || window
    //let key = Symbol()
    //obj[key] = this
    obj.fn = this                           //在该对象上添加函数
    const args = [...arguments].slice(1)    //整合函数参数
    let res = obj.fn(...args)               //调用
    delete obj.fn                           //删除增加的函数
    //delete obj[key]
    return res
}

function test(a, b) {
    console.log(this, a, b)
}

test.myCall(this, 1)
test.myCall(this, 1, 2)
test.myCall({age:12}, 1,2)

Function.prototype.myCall2 = function(thisArg, ...args) {
    if(typeof this != 'function') {
        return new TypeError('error')
    }
    thisArg = thisArg || window
    let fn = Symbol()
    thisArg[fn] = this
    let res = thisArg[fn](...args)
    delete thisArg[fn]
    return res
}
```

apply

```js
Function.prototype.myApply = function(thisArg, args){
    if(typeof this != 'function') {
        throw new TypeError(this + '不是函数')
    }
    thisArg = thisArg || window
    args = args || []      
    let key = Symbol()   
    thisArg[key] = this                      
    let res = thisArg[key](...args)               
    delete thisArg[key]                          
    return res
}
```

bind

```js
Function.prototype.myBind = function (context) {
    // 判断调用对象是否为函数
    if (typeof this !== "function") {
        throw new TypeError("Error");
    }
    // 获取参数
    const args = [...arguments].slice(1)
    const fn = this

    return function Fn() {
        if(this instanceof Fn) {    //为了兼容new调用
            return fn.apply(new fn(...arguments), args.concat(...arguments))
        }else{
            return fn.apply(context, args.concat(...arguments))
        }
        // 根据调用方式，传入不同绑定值
        // return fn.apply(this instanceof Fn ? new fn(...arguments) : context, args.concat(...arguments)); 
    }
}
```

###### 10、实现 new

```js
function Person(name, age) {
    this.name = name
    this.age = age
}
Person.prototype.sayHi = function() {
    console.log('Hi，我是'+ this.name)
}
//实现new
function create() {
    let obj = {}									//1、新建一个对象
    obj.__proto__ = Person.prototype				//2、原型链
    let args = [...arguments]						//3、获取函数参数列表
    let fn = args.shift()							//4、获取构造函数
    let res = fn.call(obj, ...args)					//5、绑定this并执行构造函数
    return typeof res === 'object' ? res : obj		//6、如构造函数返回对象则输出这个对象，若不返回对象输出obj
}
let a = create(Person, 'tom', 56)
console.log(a)
a.sayHi()

function create2() {
    let [fn, ...args] = [...arguments]
    let obj = {
        __proto__: fn.prototype
    }
    let res = fn.apply(obj, args)
    return typeof res == 'object' ? res : obj
}
```

###### 11、Promise

Promise.all

```js
//Promise.all
Promise.prototype.myAll = function(values) {
    let count = 0, n = values.length
    let res = []
    return new Promise((resolve, reject) => {
        for(let i = 0; i < n; i++) {
            Promise.resolve(values[i]).then(
                value => {
                    count++
                    res[i] = value		//不能使用res.push(value)，可能会有访问冲突
                    if(count == n) {
                        resolve(res)
                    }
                },
                error => reject(error)
            )
        }
    })
}
const promise1 = Promise.resolve(3);
const promise2 = 42;
const promise3 = new Promise((resolve, reject) => {
  setTimeout(resolve, 1000, 'foo');
})
//注意调用方式
Promise.prototype.myAll([promise1, promise2, promise3]).then(
    values => console.log(values)
)
```

Promise.race

```js
Promise.prototype.myRace = function(values) {
    return new Promise((resolve, reject) => {
        for(let p of values) {
            Promise.resolve(p).then(
                value => resolve(value),
                error => reject(error)
            )
        }
    })
}
```

###### 12、手写原生 Ajax

```js
function ajax() {
    let xhr = new XMLHttpRequest()
    xhr.open('get', 'http://httpbin.org/get')
    xhr.onreadystatechange = () => {
        if(xhr.readyState === 4) {          //xhr代理的状态，4表示下载已完成
            if(xhr.status >= 200 && xhr.status < 300) {     //HTTP状态码
                console.log(xhr.responseText)
            }
        }
    }
    xhr.send()
}
```

Promise版本

```js
function ajax(url) {
    return new Promise((resolve, reject) => {
        let xhr = new XMLHttpRequest()
        xhr.open('get', url)
        xhr.onreadystatechange = () => {
            if(xhr.readyState === 4) {         
                if(xhr.status >= 200 && xhr.status < 300) {     
                    resolve(xhr.responseText)
                }else{
                    reject('请求出错')
                }
            }
        }
        xhr.send()
    })
}
ajax('http://httpbin.org/get').then(
    value => console.log(value),
    error => console.log(error)
)
```

###### 13、节流防抖函数

节流：连续触发但 n 秒内只执行一次

```js
//方法一，setTimeout
function throttle(fn, delay) {
    let timer
    return function() {
        let args = arguments
        if(!timer) {
            fn.apply(this, args)
            timer = setTimeout(() => {
                timer = null
            }, delay)
        }
    }
}
let a = 0
function test(b) {
    a++
    console.log(a, b, '鼠标在移动')
}
let c = throttle(test, 3000) //注意，此法不能置于while循环中看效果，置于while循环中主线程宏任务永远不会停止，setTimeout也就永远不会执行
document.onmousemove = () => c('hello')     //需要函数传参的写法
// document.onmousemove = c                 //不需要函数传参的写法

//方式二，时间戳
function throttle(cb, wait=300){
    let last = 0;
    return function(){
        let now = Date.now()
        let args = arguments
        if (now - last > wait) {
            cb.apply(this, args)
            last = Date.now()
        }
    }
}
let num = 0
let a = throttle(function() {
    num++
    console.log(num, '节流函数')
}, 2000)
while(1) {
    a()
}
```

防抖：触发事件后 n 秒内只能执行一次，如果在 n 秒内又触发了事件，则会重新计算函数执行时间

```js
function debounce(fn, delay) {
    if(typeof fn != 'function') {
        throw new TypeError('fn不是函数')
    }
    let timer
    return function() {
        if(timer) {
            clearTimeout(timer)
        }
        timer = setTimeout(() => {
            fn.apply(this, arguments)
        }, delay);
    }
}
let input = document.getElementById('input')
input.addEventListener('keyup', debounce(() => console.log(input.value), 1000))
```

节流和防抖的区别

- 防抖：防止抖动（**以免把一次事件误认为多次**），单位时间内事件触发会被重置，避免事件被误伤触发多次。代码实现重在清零 clearTimeout。使用场景：登陆按钮避免用户点击太快发送多次请求、调整浏览器窗口大小、文本编辑器实时保存
- 节流：控制流量，单位时间内事件只能触发一次。代码实现重在开锁关 timer= setTimeout; timer=null。应用场景：搜索联想

###### 14、用 setTimeout 实现 setInterval

```js
//疑问：容易栈溢出？好像是不会，inner()其实就开启了一个定时器，开启完本次运行就结束了，再递归的话不会使得栈越来越大
function mySetInterval(fn, delay) {
    let inner = function() {
        let timer = setTimeout(() => {
            fn()
            clearTimeout(timer)
            inner()
        }, delay)
    }
    inner()
}
```

#### 16、JS 实现继承

###### 1、原型链继承

```js
function Parent() {
    this.name = 'parent'
    this.play = [1,2,3]
}
function Child() {
    this.type = 'child'
}
Child.prototype = new Parent()

let s1 = new Child(), s2 = new Child()
s1.play.push(4)
console.log(s2.play)    //[1,2,3,4]
```

分析

- 存在问题：s1 的属性变化会引起 s2 的属性变化

###### 2、构造函数继承

```js
function Parent() {
    this.name = 'parent'
}
Parent.prototype.getName = function() { return this.name }

function Child() {
    Parent.call(this)
    this.type = 'child'
}

let child = new Child()
console.log(child)
console.log(child.getName())    //报错
```

分析：

- 父类的引用属性不会被共享，优化了第一种继承方式的弊端
- 存在问题：只能继承父类的实例属性和方法，不能继承原型属性或者方法

###### 3、组合继承：原型链 + 构造函数

```js
function Parent() {
    this.name = 'parent'
    this.play = [1,2,3]
}
Parent.prototype.getName = function() { return this.name }

function Child() {
    Parent.call(this)   //第二次调用 Parent()
    this.type = 'child'
}
//第一次调用 Parent()
Child.prototype = new Parent()
//手动挂上构造器，否则指向的是Parent的构造函数（好像不写也没问题）
Child.prototype.constructor = Child     

let s1 = new Child(), s2 = new Child()
s1.play.push(4)
console.log(s1, s2)
console.log(s1.play, s2.play)
console.log(s1.getName(), s2.getName())
```

分析

- 解决了前两种方式存在的问题，但是Parent() 执行了两次，多了一次构造的性能开销

###### 4、原型式继承

```js
//主要借助Object.create方法
let parent4 = {
    name: "parent4",
    friends: ["p1", "p2", "p3"],
    getName: function() {
      return this.name;
    }
  };

  let person4 = Object.create(parent4);
  person4.name = "tom";
  person4.friends.push("jerry");

  let person5 = Object.create(parent4);
  person5.friends.push("lucy");

  console.log(person4.name); // tom
  console.log(person4.name === person4.getName()); // true
  console.log(person5.name); // parent4
  console.log(person4.friends); // ["p1", "p2", "p3","jerry","lucy"]
  console.log(person5.friends); // ["p1", "p2", "p3","jerry","lucy"]
```

分析

- 存在问题：该方法实现的是浅拷贝，引用类型属性容易被篡改

###### 5、寄生式继承

```js
//没看明白这种方法
let parent5 = {
    name: "parent5",
    friends: ["p1", "p2", "p3"],
    getName: function() {
        return this.name;
    }
};

function clone(original) {
    let clone = Object.create(original);
    clone.getFriends = function() {
        return this.friends;
    };
    return clone;
}

let person5 = clone(parent5);

console.log(person5.getName()); // parent5
console.log(person5.getFriends()); // ["p1", "p2", "p3"]
```

###### 6、寄生组合式继承

```js
//最优解
function Parent() {
    this.name = 'parent'
    this.play = [1,2,3]
}
Parent.prototype.getName = function() { return this.name }

function Child() {
    Parent.call(this)
    this.friends = 'child'
}
//好像就是把组合继承中的 Child.prototype = new Parent() 改成了下面这句话
Child.prototype = Object.create(Parent.prototype) 

Child.prototype.constructor = Child
Child.prototype.getFriends = function() { return this.friends }

let p = new Child()
console.log(p)
console.log(p.getName())
console.log(p.getFriends())
```

#### 17、typeof 和 instanceof

###### typeof

- typeof 操作符返回一个字符串，表示未经计算的操作数的类型
- typeof 的返回结果为 number、string、undefined、boolen、symbol、object、function 的其中之一
- 有一个特殊的地方：typeof(null) 结果为 object
- 判断变量是否存在可以使用`typeof a != 'undefined'`

###### instanceof

- instanceof 运算符用于检测构造函数的 prototype 属性是否出现在某个实例对象的原型链上

###### 区别

- typeof 会返回一个变量的基本类型，instanceof 返回的是一个布尔值

- instanceof 可以准确地判断复杂引用数据类型，但是不能正确判断基础数据类型 

  ```js
  let a = 1
  console.log(a instanceof Number)    //false
  ```

- 而 typeof 也存在弊端，它虽然可以判断基础数据类型（null 除外），但是引用数据类型中，除了function 类型以外，其他的也无法判断

- 如果需要通用检测数据类型，可以采用Object.prototype.toString，调用该方法，统一返回格式“[object Xxx]”的字符串

  ```js
  function getType(obj){
    if(obj === null) return null
    let type  = typeof obj;
    if (type !== "object") {    // 先进行typeof判断，如果是基础数据类型，直接返回
      return type;
    }
    // 对于typeof返回结果是object的，再进行如下的判断，正则返回结果
    return Object.prototype.toString.call(obj).replace(/^\[object (\S+)\]$/, '$1'); 
  }
  ```


#### 18、尾递归

###### 正常递归

```js
function factorial(n) {
  if (n === 1) return 1
  return n * factorial(n - 1)
}
```

###### 尾递归

```js
function factorial(n, total) {
    if (n == 1) return total
    return factorial(n-1, n*total)
}
```

###### 区别

- 正常递归调用栈为 O(n)，容易溢出，而尾递归调用栈为 O(1)

#### 19、内存泄漏

###### 垃圾回收机制

- 标记清除
- 引用计数（优化方法，用不到的引用及时设置为 null，如对象、定时器、DOM引用）

#### 20、隐式类型和比较

###### 1、条件判断时：自动转为布尔值

`null、undefined、0、空字符串、false、NaN`转为 false，其余转为 true

```js
//注意
Boolean([])		//true
Boolean({})		//true
```

###### 2、遇到运算符 + 时：常常转为字符串

- 遇到预期为字符串的地方，就会将非字符串的值自动转为字符串

- 具体规则是：先将复合类型的值转为原始类型的值，再将原始类型的值转为字符串

- 举例：字符串+数字、数字+对象/函数、函数+函数

###### 3、遇到运算符 - 时：常常自动转为数字

```js
'5' - '2' 		// 3
'5' * '2' 		// 10
true - 1  		// 0
false - 1 		// -1
'1' - 1   		// 0
'5' * []    	// 0
false / '5' 	// 0
'abc' - 1   	// NaN
null + 1 		// 1
undefined + 1 	// NaN
```

注意：null 转为数值时，值为0 ；undefined 转为数值时，值为NaN

###### 4、各种比较

- == 比较规则
  - 两个都为简单类型，字符串和布尔值都会转换成数值，再比较
  - 简单类型与引用类型比较，对象先调用 toString() 方法得到字符串 str，然后再将 str 转为数值类型，即 Number(str)，之后再参与比较
  - 两个都为引用类型，则比较它们是否指向同一个对象
  - null 和 undefined 相等
  - 存在 NaN 则返回 false
  
  ```js
  //对于第二条规则的验证
  let a = []
  console.log(a == false)	//true，因为 a.toString() = ''，Number('') = 0
  //如果修改toString()方法
  a.toString = function() { return 'daas' }
  console.log(a == false)	//false
  ```
  
- === 比较规则
  - 只有两个操作数在不转换的前提下相等才返回 `true`。即类型相同，值也需相同

```js
null == undefined  	//true
null === undefined //false

NaN == NaN   //false
NaN === NaN  //false

[] == ![]   //true，前者为Object，后者为boolean且值为false，分析见下面
[] == !{} 	//false,前者为Object，后者为boolean且值为false，分析见下面
[] == false	//true，对象VS基本类型，Number([].toString()) = Number('') = 0，Number(false) = 0，0 == 0，true
{} == false	//false，对象VS基本类型，Number({}.toString()) = Number('[object object]') = NaN，NaN == 0，false

'hello' == new String('hello')		//true
'hello' ===  new String('hello')	//false

//有几个组的比较需要注意
'' == '0' 	// false
0 == '' 	// true，转化为：0 == 0
0 == '0' 	// true，转化为：0 == 0

false == 'false' 	// false，转化为：0 == NaN
false == '0' 		// true，转化为：0 == 0

//以下全为 false
null == true
null == false
null == 0
undefined == true
undefined == false
undefined == 0

' \t\r\n' == 0 		// true
```

#### 21、JS 修改 DOM 样式

> 修改的样式属性必须使用内联样式，即 xxx.style.xxx

```js
div.width = '200px'			//没用
div.style.width = '200px'	//正确操作
let width = getComputedStyle(div).width		//获取 width 属性值
```

