<h1 style="text-align:center;">CSS</h1>

#### 1、input:checkbox

- 两个属性 defaultChecked 和 checked，对比如下：
  - defaultChecked：可读写，其状态不会随着其余相关联组件而变化，只在第一次显示时奏效，如果不与其他相关联，用这个属性即可
  - checked：只读，如果定义 onChange 事件，可实现读写，且状态会随着其余相关联组件而变化
- 取得 checkbox 的值，不是 event.target.value 或者 ref.current.value，而是 event.target.checked 或者 ref.current.checked

#### 2、选择器使用准则
- CSS使用class，js使用id
  
- CSS
  
  1. 尽量使用class，可定义一些原子类用于定义公共样式
  2. 尽量（一定）不要用 **单独** 用标签选择，多个文件时，用标签写的样式会存在覆盖问题
  3. 上线网站不要使用通配符（*），效率低，平常书写随意
  4. 标签选择器无视父子关系选择所有，类选择相同类名元素，id选择特定id元素，当使用复杂选择器时，比如后代选择器，仍然是符合这些标准的
  5. 元素含有多个类名，class="one two"，若要匹配所有类名，选择方式为：.one.two（连续点，无序）
  6. 注意选择器优先级问题：
     ```html
     <div class="mydiv">
             <p class="div-p"></p>
             <p class="div-p"></p>
             <p class="div-p"></p>
             <p class="div-p"></p>
         </div>
     ```
     .mydiv p { } 的优先级会高于 .div-p { } 的优先级，类似情况学会自己分析
     **注意：选择子元素时，将其父元素也写在前面，保证其优先级够高，避免单独写子元素，导致样式被层叠掉**
  7. 选择器应用问题
     A、B是兄弟节点，实现悬浮A时，B显示：A:hover + B
     A是B的父节点，实现悬浮A时，B显示：A:hover B

#### 3、CSS3新增选择器

- 子级选择器（>），区别于后代选择器，直接子级

- 相邻兄弟选择器（E1+E2），E2紧跟在E1后面

- 其他兄弟选择器（E1~E2），匹配在E1后面的所有E2元素

- 结构选择器

  - :first-child，**有三种写法**，类似的还有:last-child、:nth-child(n)，**注意：这里的child应该是直接子级**

    ```css
    .box:first-child	//不带空格，表示选中body（默认）中的第一个box（整个盒子）
    .box :first-child	//带空格，表示选中box的第一个子元素
    .box p:first-child 	//带空格且带标签，表示选中box的第一个子元素，且该子元素是p标签，若第一个子元素不是p标签，则未选中，这里的p标签也可写成class、id等
    ```

  - :nth-child(n)中n的写法，可写数字、关键字、表达式

    - 数字代表第几个
    - 关键字有 even 和 odd
    - 表达式为 n 的表达式，n取值范围：0、1、2、3...直至表达式值不满足条件为止，如(-n+5)代表选择前五个元素

  - :first-of-type，**有三种写法**，类似的还有:last-of-type、:nth-of-type(n)

  ```css
      .box:fisrt-of-type		//不带空格，选中第一个box
      .box :first-of-type		//带空格，选中box中所有类型元素的第一个，比如第一个h1标签，第一个p标签......
      .box p:first-of-type	//带空格带标签，选中box中第一个p标签
  ```

- 伪元素选择器
  - E::before，在E元素内部的前面插入一个元素，即第一个儿子（单冒号，双冒号均可，双冒号是h5的写法）
  - E::after，在E元素内部的后面插入一个元素，即最后一个儿子
  - E::first-letter，选中E容器内的第一个字母，一般对这个字母做放大等处理
  - E::first-line，选中E容器内的第一行文本
  - ::前不允许有空格，before和after伪元素内部必须写上`content:""`属性，否则不可见，其新增的元素为行内元素
- 属性选择器
  - E[att]：选择具有att属性的E元素
  - E[att = "val"]：选择具有att属性且值为val的E元素
  - E[att ^= "val"]：选择具有att属性且值以val开头的E元素
  - E[att $= "val"]：选择具有att属性且值以val结尾的E元素
  - E[att *= "val"]：选择具有att属性且值中含有val的E元素

#### 4、选择器权重（层叠性中的优先级比较）

> 比较原则：距离、权重、书写顺序

- 先比较距离，即样式写在谁身上
  - 最近的就是目标节点自身，再依次是目标节点的父节点、爷节点等祖先节点（继承）
- 距离相同时，比较权重
    - 基础选择器：选择范围越大权重越小，* < 标签 < class < id，另外伪类选择器、属性选择器的权重等于类选择器；伪元素选择器的权重等于标签选择器
    - 高级选择器：依次比较组成高级选择器的 id、class、标签的个数，如果前面能够比较出大小就不用再比较后面
    - !important 可以将**某条CSS属性**的权重设置为最大，```background-color: green !important;```
    - 行内式样式权重高于各种选择器，但是低于!important

- 距离和权重都相同时，比较书写顺序

#### 5、标准文档流和脱标

- 微观现象

  - 空白折叠：编辑时标签与标签之间的空格、换行都会被压缩成一个空格
  - 文字类元素排在一行会出现高低不齐、底部对齐的情况，由此引出属性```vertical-align: baseline;```，可设置对齐方式
  - 自动换行

- 元素等级

  > 块级元素、行内元素、行内块元素

  - 块级元素：大部分容器级标签，div、h1、p等
    - 可设置宽高
    - 独占一行
    - 块级元素宽度默认值为父元素宽度的100%，高度不设置时会被内容自动撑开
  - 行内元素：大部分文本级标签，span、a等
    - 不能正常设置宽高，其他盒模型属性设置后会出现加载问题
    - 可与其他行内或者行内块并排显示
    - 宽高只能被内容自动撑开
  - 行内块元素：img、input等
  - 
    - 可设置宽高
    - 可与其他行内或者行内块并排显示
    - 若不设置宽高，会以原始尺寸（图片尺寸、空按钮默认大小等）或者被内容自动撑开
    - **行内块依旧具有标准流的3个微观性质**，注意微观性质中的第2条，就可以解释将一堆高度不一致的div设置为inline-block会出现高低不齐的效果，它会按照div中内容的底部（默认为底部）进行对齐排列，如果div为空，底部即为自身高度

- 脱标

  - 浮动、绝对定位、固定定位可以脱标
  - 脱标的元素具备行块二象形，可以设置宽高，可以并排一行，且**不会有空白折叠现象和margin塌陷现象**
  - 当父元素未设置宽高时，考察能否被子元素自动撑开，共四种情况
    - 父标子标：可以撑开
    - 父标子脱：不能撑开
    - 父脱子标：可以撑开
    - 父脱子脱：可以撑开
#### 6、浮动

- 性质
  - 具备脱标元素的性质
  - 浮动的元素依次贴边，以left为例，所有子元素依次向左贴边，子元素1会贴到父元素左边，子元素2会贴到子元素1上，子元素3会贴到子元素2上......若父元素中剩余宽度不够，放不下子元素3，子元素3会跳过子元素2往子元素1上贴，再不够就往父元素左边贴，如果再不够，还是会贴在父元素左边，且出现溢出效果
  - 元素浮动后会让出原标准流位置给下一个标准流元素，且会压盖在那个标准流元素上
  - 字围效果，同级元素A、B，A浮动，B不浮动，B中的文字就会围绕A分布，但除去特殊需求，一般不允许同级元素有的浮动有的不浮动
  
- 浮动存在的问题

  - 浮动导致脱标，使得在父元素不设置高度的情况下，子元素撑不起标准流中父元素的高度
  - 父元素没有高度，会影响后面元素的标准流位置，如果浮动的子元素足够高，后面的浮动元素就会挨着那个子元素贴边

- 清除浮动影响的几种方法

  - 给标准流的父元素设置height：定死了，不够灵活
  - 外墙法、内墙法：分别在父元素外面和里面添加空标签，用于清除浮动，其中外墙法没有解决父元素高度自适应，内墙法解决了，但这属于使用HTML来消除CSS的影响，本末倒置，且多添加了很多标签

  - 伪类选择器（类名一般取为clearfix），与内墙法机制一样

    ```css
    .clearfix::after {   /*相当于在父元素内部的最后面加一个元素，即最后一个虚拟子元素*/
      content: "";    /*新添加的元素内容为空*/
      clear: both;    /*清除浮动*/
      display: table; /*其余块级元素也可以，可能需要补上height:0、visibility:hidden等属性*/
    }
    
    .clearfix::after {  
      content: ""; 
      display: block;
      height: 0;
      clear: both;   
      visibility:hidden;
    }
    ```
  
  - 父元素添加overflow: hidden; ：形成BFC，使用前提是没有定死height，这时父元素会寻找子元素中的最大高度（不论子元素是否浮动）作为自己的高度，实现高度自适应的效果

#### 7、position

- 相对定位relative

  - 参考元素：自身的原始位置

  - 性质：相对定位的元素不脱离标签的原始状态（标准流、浮动），不会让出原来占有的位置
  - 移动规则：left、right等定义的是偏移量，left: 10px表示距离其原始位置的左边有10px的偏移量，相当于向右移动了10px
  - 相对定位元素比较稳定，不会随意让出位置，可将相对定位的元素作为后期绝对定位的参考元素，即“子绝父相”
  - 应用：导航条顶部添加border，同时保证导航条不下移；文字上下标；居中显示

- 绝对定位absolute
  - 参考元素：是距离最近的有定位的祖先元素，**若祖先均没有定位，参考浏览器窗口（不是body，不是html）**，注意：position的默认值为static，是无定位状态，即标准流，其余属性为已定位状态，就是找到一个position为非默认值的最近祖先元素
  - 性质：脱离标准流，让出原先的位置
  - 移动规则：参考点不是固定的，当定义left和top时，参考点为祖先元素的左上顶点；当定义right和bottom时，参考点为祖先元素的右下顶点（**注意此处的顶点指的是祖先元素带padding区域的顶点，也就表明元素可以移动到祖先元素的padding区域里**）
  - 应用：压盖效果；居中显示
- 固定定位fixed
  - 参考元素：浏览器窗口
  - 参考点：浏览器窗口的4个顶点，与绝对定位absolute类似，具体是哪个顶点与偏移量组合方向有关
  - 性质：脱离标准流，让出原先的位置
- 相对定位未脱标，绝对定位和固定定位已脱标（此外浮动也会脱标），**脱标之后的元素可以正常设置宽高**
- position和float可以同时使用，float贴边还是参考上一个浮动元素的原本位置而不是移动后的位置，position的参照元素不变，但有可能出现absolute和fixed覆盖掉float效果的情况

#### 8、居中显示
1. 文本垂直居中
   - 单行文本：设置line-height等于父元素高度
   - 多行文本；父元素高度自适应，上下padding值设置成一样
2. 元素垂直居中：父元素高度自适应，设置相同上下padding值
   元素水平居中：margin: 0 auto，只适用于父元素宽度大于子元素宽度的情况
   注意：auto只适应于水平方向上，原理为：margin-left: auto 会将元素一直往右挤至父元素右边界，而margin-right: auto 会将元素往左挤，两个方向同时设置auto，就会达到平衡，使元素居中
3. 使用 position 和 margin 实现水平居中：第一步：left: 50%; 第二步：margin-left: -100px，这里的100为元素宽度的一半，总结就是：往右走上父元素宽度的一半，再往左回自身宽度的一半，这种方法对于父元素宽度小于子元素宽度的情况同样适用；同样垂直居中也可以使用这种方法实现
   补充：第二步margin可以用transform: translate(-50%)代替，表示向左移动自身宽度的一半
4. flex、grid 布局
   水平居中：justify-content: center; 
   垂直居中：align-items: center;
5. 有一类特殊的：背景图像中可直接使用：background-position: 50% 50%;这里的百分比是背景区域减去背景图像大小，作为背景图像移动距离

#### 9、压盖顺序

- 默认压盖顺序
  - 定位元素不区分定位类型，都会区压盖标准流或者浮动元素
  - 如果都是定位元素，按照书写顺序来定
- 自定义压盖顺序（z-index）
  - 属性值大的会压盖属性值小的，设置z-index属性的会压盖没有设置的
  - 属性值相同时比较书写顺序
  - **z-index只对已定位元素生效**
  - 父子盒模型中，若父子均已定位，与其他父子级有压盖的部分：
    - 父级盒子：若不设置z-index， 按照书写顺序，否则按照值大小；
    - 子级盒子：若父级没有z-index，子级z-index大值压小值，若父级设置z-index，一切按照父级属性值大小，俗称“从父效应”
    - 总之就是：父与父比试，胜者显示；子与子比试，胜者显示，但子与子比试时可以拼爹

#### 10、伪类/伪元素

- 伪类指：:hover、:first-child等，伪类用于选择DOM树之外的信息，或是不能用简单选择器进行表示的信息
- 伪元素指：::before、::after、::first-letter、::first-line等，伪元素为DOM树没有定义的虚拟元素。不同于其他选择器，它**不以元素为最小选择单元**，它选择的是元素指定内容
- 以CSS3的语法来讲，伪类是单冒号，伪元素是双冒号

#### 11、过渡属性transition

> 当某属性发生变化时，默认是瞬间完成这个变化过程，transition是给这个变化过程加了一个过渡，使之有一个缓慢变化的视觉效果

- transition: 监视的属性（一般写all）、过渡时间、时间曲线、延迟时间（一般是立即执行，写0s，单位不能丢）

  ```css
  .box {
      height: 100px;
      width: 100px;
      transition: all 2s linear 0s;
      -webkit-transition: all 2s linear 0s;	//解决在Safari浏览器的兼容问题
  }
  .box:hover {
      width: 500px;	//发生了变化，当鼠标移开后会逆着恢复原样
  }
  ```

- 时间曲线cubic-bezier()，cubic-bezier(x1,y1,x2,y2)中传入两个点的x、y坐标，范围[0,1]，另外有两个默认点：起始点[0,0]，结束点[1,1]，这四个点连成一条运动曲线，且该曲线被划分为3段，分别对应开始、中间、结束时的运动情况，linear等是cubic-bezier()的一些默认值

- transition: width 2s linear 0s, height 4s linear 0s; 还可以写多个值，逗号隔开

#### 12、CSS中属性值正负问题

> 一切参考坐标轴，CSS中，坐标原点为窗口左上角，从左到右为X正轴，从上往下为Y正轴，正值沿正轴，负值沿负轴

- ```transform: translate(100px, 200px);```是往右走了100，往下走了200
- position同样是这种分析方法

#### 13、CSS中不是所有的属性都是可以继承的
- 比如background-color就是不可继承的，在浏览器控制台查看时，父元素属性一栏中颜色为灰色的属性就是不可继承的，另外属性被划掉说明被层叠掉了
- 能够被继承的只有**文字相关的属性**，其他的都不能被继承
- 浏览器控制面板中的样式分成了很多块，还是按照权重排列的，最上面的 element.style 是行内样式，接着是内嵌样式/外联样式、用户代理样式表（浏览器默认设置）、父类继承

#### 14、margin 塌陷

- 兄弟元素塌陷
  - 将margin设置到一个元素身上，不要分开设置到两个元素上，避免相遇

- 父子元素塌陷，父元素margin-top为40，子元素margin-top为50，则父元素的margin-top会塌陷到子元素里面，父子元素之间没有margin，父子整体作为大盒子的margin-top为50；特殊的当父元素margin-top为0，子元素margin-top为50时，会出现子元素带着父元素掉下来的情况，解决办法就是不让两个margin相遇，用父元素的border或者padding隔开
  - 父元素上设置border
  - 用父元素的padding代替子元素的margin
- 注意：margin塌陷只会出现在垂直方向上，水平方向上不会出现

#### 15、覆盖区域

- background-color的覆盖区域包括border、padding和本体
- background-image对于repeat图片的覆盖为border及以内，对于no-repeat图片的覆盖为border以内

#### 16、背景图应用

- 网站logo：将logo图作为h1>a的背景图，再将文字隐藏（通过 text-indent: -999em 和 overflow: hidden）
- padding区域背景图
- 精灵图
- 注意：多个背景图，先写的会压盖在后写的上面

#### 17、a标签的四个伪类属性

1. 伪类权重一样，只能按照书写顺序层叠，要实现正常效果，书写顺序为：
   link、visited、hover、active（点击前、点击后、鼠标悬浮、鼠标正在点击未松开）
2. 可以选择a标签，定义公共属性（包括以上四种伪类），然后将a:hover单独拎出来定义悬浮属性进行重叠

#### 18、重排（回流）、重汇

> 以盖房子为例，将DOM元素看成一块块砖头

- 重排（回流）：把很多块砖头垒成墙，当其他砖头位置变化时，会触发重排；另外获取一些需要计算才能得到的属性时，也会触发重排（如offsetTop、offsetLeft）
- 重汇：只改变其中某块砖头的某些样式（如颜色），但不因影响其他砖头的位置，会触发重汇
- 重排一定有重汇，重汇不一定有重排

#### 19、display: none VS visibility: hidden VS opacity: 0

|                                     | opacity: 0 | visibility: hidden | display: none |
| ----------------------------------- | :--------: | :----------------: | :-----------: |
| 是否占据页面空间                    |    占据    |        占据        |    不占据     |
| 子元素是否可以自由控制显示与否      |   不可以   |        可以        |    不可以     |
| 自身绑定的事件能否继续触发          |     能     |        不能        |     不能      |
| 是否影响遮挡住的元素触发事件        |    影响    |       不影响       |    不影响     |
| 是否有回流（重排）                  |    没有    |        没有        |      有       |
| 是否有重绘                          |   不一定   |         有         |      有       |
| 是否支持transition（缓慢出现/消失） |    支持    |        支持        |    不支持     |

#### 20、BFC

> 块格式化上下文

- 定义：它是一个只有块级盒子参与的独立块级渲染区域，它规定了内部的块级盒子如何布局，且与区域外部无关；BFC 就是页面上的一个隔离的独立容器，容器里面的子元素不会影响到外面的元素，反之亦然
- 如何实现BFC：
  - html 根元素
  - float: left、right
  - position: absolute、fixed
  - display: inline-block、inltable-cell、table-caption、table、inline-table、flex、inline-flex、grid、inline-grid
  - overflow: auto、hidden、scoll（即除去默认值 visible）
- BFC的作用
  - overflow: hidden：清除浮动影响，使父元素包裹住浮动的子元素，实现高度自适应
  - overflow: auto;：也可以清除浮动影响，但有时会出现滚动条，带来副作用，可在父元素上使用 display: flow-root
  - 消除margin塌陷：这就是float元素间不存在margin塌陷的原因
  - BFC 的区域不会与 float 的元素区域重叠，因此可以用来清除字围效果

#### 21、常用单位

- 绝对单位：px
  - 将屏幕看成一个个的点，10px 固定占据10个点
- 相对单位：em、rem、vw、vh
  - em 所谓的相对是相对于当前元素的`font-size`属性值大小，注意该属性是可以继承的，若当前元素未设置，会继承其父元素的属性值大小，例如 A > B，A中设置`font-size:10px`，B中未设置，那么B中的 1em 就是 10px，1.5em 就是 15px
  - rem 与 rm 类似，但永远以`html`中的`font-size`属性值大小为参考
  - vw、vh 即窗口宽度、高度，分成100等份，50vm 表示一般的窗口宽度
  - 另 vmax、vmin 分别表示 vw、vh 中的最大值、最小值
