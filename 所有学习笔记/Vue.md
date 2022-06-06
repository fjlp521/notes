<h1 style="text-align:center;">Vue</h1>

##### 1、Vue的特点

- 组件化，复用率高，易维护
- 声明式编码，无需直接操作DOM
- 虚拟DOM + Diff算法，尽量复用DOM节点

##### 2、Vue实例与容器的关系

- 一对一，不是多对一，也不是一对多

##### 3、v-model 

- 只能用于表单类元素（输入型元素），这些元素基本上都有 value 属性，v-model:value = "xxx"，也可以简写为 v-model = "xxx"，v-model 默认收集的就是 value 值

##### 4、Vue对原始数据的处理：数据劫持和数据代理

- 第一步：数据劫持，加工原始数据使其具有响应式，即可以实现对数据的监测（vue中的函数名字：reactiveGetter，reactiveSetter）
- 第二步：数据代理，将加工后的数据直接挂到Vue实例身上，便于操作（vue中的函数名字：proxyGetter，proxySetter）

###### vue2版：Object.defineProperty

  ```js
  //原始数据 
  let data = {
      name:'tom',
      age: 15
  }
  
  //数据劫持（简易版本，对于复杂数据，如对象里面包含对象，数组里面包含对象等，还需要递归实现）
  function Observe(data) {
      let keys = Object.keys(data)
      keys.forEach((k)=>{
          Object.defineProperty(this, k, {
              enumerable: true,       //可枚举，保证Object.keys()可以读取到    
              // configurable: true,
              get: function reactiveGetter() {
                  console.log('数据劫持：reactiveGetter被调用')
                  return data[k]
              },
              set: function reactiveSetter(val) {
                  console.log('数据劫持：reactiveSetter被调用')
                  data[k] = val
                  //然后重新渲染模板
              },
          })
      })
  }
  let obj = new Observe(data)
  let vm = {}
  vm._data = data = obj
  
  //数据代理
  Object.keys(vm._data).forEach((k)=> {
      Object.defineProperty(vm, k, {
          get: function proxyGetter(){
              console.log('数据代理：proxyGetter被调用')
              return vm._data[k]
          },
          set: function proxySetter(val) {
              console.log('数据代理：proxySetter被调用')
              vm._data[k] = val
          },
      })
  })
  ```

###### vue3版：Proxy，Reflect

```js
//原始数据 
let data = {
    name:'tom',
    age: 15,
    books: ['AA', 'BB', 'CC']
}

let vm = {}
//无需循环递归，只写get、set就行
vm._data = new Proxy(data,{
    get(target, props) {
        console.log('get')
        return Reflect.get(target, props)
    },
    set(target, props, value) {
        console.log('set')
        Reflect.set(target, props, value)
    }
})

Reflect.ownKeys(vm._data).forEach((k) => {
    Reflect.defineProperty(vm, k, {
        get() {
            return vm._data[k]
        },
        set(val) {
            vm._data[k] = val
        }
    })
})
```

##### 5、Vue 中数据监测的程度

- 对象的监测：对象中所有属性都生成了 getter 和 setter，因此可以监测到对象内部属性的改变

- 数组的监测：只有使用了特定的7个方法（shift、unshift...），才能监测到变化，这些方法经过了 vue 的封装，可以进行监测

  ```js
  new Vue({
      el:'#root',
      data:{
          persons:[
              {id:'001',name:'马冬梅',age:30,sex:'女'},
              {id:'002',name:'周冬雨',age:31,sex:'女'},
              {id:'003',name:'周杰伦',age:18,sex:'男'},
              {id:'004',name:'温兆伦',age:19,sex:'男'}
          ]
      },
      methods: {
          updateMei(){
              // this.persons[0].name = '马老师'       //奏效
              // this.persons[0].age = 50             //奏效
              // this.persons[0].sex = '男'           //奏效
              // this.persons[0] = {id:'001',name:'马老师',age:50,sex:'男'}       //不奏效
              this.persons.splice(0,1,{id:'001',name:'马老师',age:50,sex:'男'})   //奏效
          }
      }
  }) 
  ```

- 如需给后添加的属性做响应式，请使用如下API：

  ​                          Vue.set(target，propertyName/index，value) 或 

  ​                          vm.$set(target，propertyName/index，value)

##### 6、methods VS computed VS watch

  - methods 无缓冲，computed 和 watch 有缓存（只在初次读取和所依赖的数据变化时才调用，比如连读5次，只会调用1次）

  - methods 和 computed 依赖于函数返回值，无法开启异步任务，watch 不依赖函数返回值，可以开启异步任务

  - methods 和 computed 写法一样（调用时不一样），只是 computed 多了缓存机制；computed 能实现的，watch 一定可以实现，反之则不行

  - watch 的代码会比较复杂，故一般能用 computed 实现的，优先选择 computed

    ```js
    		//<span>{{fullname()}}</span>
    		methods: {
                fullname() {
                    return this.firstname + '-' + this.lastname
                }
            },
                
            //<span>{{fullname}}</span>
            computed: {
                fullname() {
                    return this.firstname + '-' + this.lastname
                }
            },
            //<span>{{fullname}}</span>，且data中需要提前定义fullname属性
            watch: {
                firstname(val){
                    this.fullname = val + '-' + this.lastname
                },
                lastname(val) {
                    this.fullname = this.firstname + '-' + val
                }
            }
    ```
    
- 对于要捕捉数据的变化，做些更新页面之外的事情，用 watch 比较合适，例如 todos 案例中监视 todos 的变化，及时更新到 localStorage 中

  ```js
  watch: {
      todos: {
        deep: true,	//开启深度监视，监测数组内部元素中的变化
        handler(newValue) {
          localStorage.setItem('todos', JSON.stringify(newValue))
        }
      }
    }
  ```

##### 7、Vue 中的 this

  - 所有被 vue 管理的函数（methods、watch等）写成普通函数，这样函数内部的 this 指向 vm 或组件实例对象
  - 所有不被 vue 管理的函数写成箭头函数（定时器回调，ajax回调等各种回调函数），箭头函数没有自己的 this，就会往外找，就找到了 vm 或组件实例对象

##### 8、vue 事件绑定
- 可以写简单语句

  ```html
  <div @click="ishot = !ishot"></div>
  ```

  注意：里面只能写 vue 管理的东西，比如写 alert(1) 是没有效果的

##### 9、绑定样式

  > 确定的样式直接写，不确定的样式通过指令绑定 class 或者 style 实现

  - class样式：写法：class="xxx"，xxx可以是字符串、对象、数组
    - 字符串：绑定一个样式，类名不确定，要动态获取
    - 对象：绑定多个样式，个数不确定，名字也不确定
    - 数组：绑定多个样式，个数确定，名字确定，但不确定用不用
  - style样式
    - 对象：:style="{fontSize: xxx}"其中xxx是动态值
    - 数组：:style="[a,b]"其中a、b是样式对象。

##### 10、v-show VS v-if

  - v-show 底层通过修改 display 属性实现，节点一直存在，适用于切换频率高的场景
  - v-if 底层通过增删节点实现，代价较大，适用于切换频率低的场景
  - v-if 可以配合 <template v-if="true"></template>>使用，v-show不行
  - v-if 会增删节点，某节点删除后就无法获取到，但 v-show 一直可以获取到

##### 11、收集表单数据
- input:type=text，v-model收集的是value值，用户输入即为value值

- input:type=radio，v-model收集的是value值，**且要给标签配置value值**

    ```js
    <input type="radio" name="sex" v-model="sex" value="female">女
    <input type="radio" name="sex" v-model="sex" value="male">男
    data: {
        sex: 'female'
    }
    ```

- input:type=checkbox，若没有配置value，收集的是checked（bool），若配置value，分两种情况，v-model的初始值是非数组，收集的是checked，v-model的初始值是数组，收集的是value组成的数组

    ```js
     爱好<br>
     <input type="checkbox" v-model="hobby" value="study">学习
     <input type="checkbox" v-model="hobby" value="eat">吃东西
     <input type="checkbox" v-model="hobby" value="game">游戏
    data: {
        hobby: '',	//没有value或者有value但hobby不是数组，收集的都是checked（bool）
        hobby: [],	//有value，收集的为value组成的数组
    }
    ```

- v-model有三个修饰符
    - v-model.lazy，失去焦点才会收集数据
    - v-model.number，输入字符串转为有效数字
    - v-model.trim，去除输入值前后的空格

##### 12、Vue 生命周期

- 共4对，8个：beforeCreate、created；beforeMount、mounted；beforeUpdate、updated；beforeDestory、destoryed
- beforeMount：页面呈现的是未经vue编译的DOM结构，所有对DOM的操作，最终都不奏效
- mounted：页面呈现的是经过vue编译的DOM结构，一般在此进行**开启定时器、发送网络请求、订阅消息、绑定自定义事件等初始化操作**
- beforeUpdate：此时数据是新的，但页面是旧的，未保持同步
- updated：数据是新的，页面也是新的，保持同步
- beforeDestoryed：vm中所有的data、methods、指令等均可用，但修改数据不再引起页面更新，一般在此进行**关闭定时器、取消订阅消息、解绑自定义事件等收尾工作**
- vue实例销毁后自定义事件会失效，但原生DOM事件依然有效（如click等）

##### 13、组件中的data写成函数式

- data必须写成函数，为什么？ ———— 避免组件被复用时，数据存在引用关系

##### 14、ref属性

- 被用来给元素或子组件注册引用信息（id的替代者）

- 应用在html标签上获取的是真实DOM元素，应用在组件标签上是组件实例对象（vc）

- 使用方式：

  - 打标识：```<h1 ref="xxx">.....</h1>``` 或 ```<School ref="xxx"></School>```
- 获取：```this.$refs.xxx```

##### 15、props配置项

- 功能：让组件接收外部传过来的数据

- 传递数据：```<Demo name="xxx"/>```

- 接收数据：
  - 第一种方式（只接收）：```props:['name'] ```
  
  - 第二种方式（限制类型）：```props:{name:String}```
  
  - 第三种方式（限制类型、限制必要性、指定默认值）
  
    ```js
    props:{
        name:{
            type:String, //类型
            required:true, //必要性
            default:'老王' //默认值
        }
    }
    ```
  
- 备注：props是只读的，Vue底层会监测你对props的修改，如果进行了修改，就会发出警告，若业务需求确实需要修改，那么请复制props的内容到data中一份，然后去修改data中的数据

##### 16、Vue 组件间通信

- props
  - 父传子：直接传数据即可
  - 子传父：父组件传递给子组件一个函数，子组件调用函数时传入的参数就会被父组件接收到
  
- 自定义事件：适用于**子传父**

- ref

  ```html
  <!--父组件，获取数据 this.$refs.foo-->
  <Children ref="foo" />  
  ```

- 全局事件总线：适用于任意组件间通信

- 消息订阅与发布

- slot插槽：父传子

- vuex：适用于多组件共享数据

##### 17、slot插槽

> 父组件向子组件传递标签体内容（props传递的是标签属性）

- 数据在父组件中，可以使用默认插槽和具名插槽，默认插槽没有名字，具名插槽有名字，可以根据名字灵活指定
- 数据在子组件中，需使用作用域插槽，使得子组件可以给父组件传递数据内容，父组件可以控制数据展示的样式
