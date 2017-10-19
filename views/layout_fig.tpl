% include header title=title
  <body>
    <div class="body">
        <h1>{{title}}</h1>
            <p>
             % for f in figures :
             <img src="/static/{{f}}" alt="Figure" width=75%><br>
             % end
            </p>
    </div>
  </body>
</html>
